"""Saved-state LV arithmetic diagnosis: no fit, optimizer update, or OpenMM."""
import argparse
import json
import math
from pathlib import Path
import torch


def json_safe(value):
    """Keep nonfinite diagnostic evidence explicit in strict JSON."""
    if isinstance(value,float) and not math.isfinite(value):
        return 'nan' if math.isnan(value) else ('+inf' if value>0 else '-inf')
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    return value


def scalar_audit(value):
    v=value.detach()
    return dict(finite=bool(torch.isfinite(v).all()),max_abs=float(v.abs().max()))


def variance_audit(value):
    return dict(input=scalar_audit(value),fp32=scalar_audit(value.var(unbiased=True)),
                fp64=scalar_audit(value.double().var(unbiased=True)))


def gradient_audit(model):
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    return dict(elements=sum(g.numel() for g in gradients),
        nonfinite_elements=sum(int((~torch.isfinite(g)).sum()) for g in gradients),
        norm32=float(torch.sqrt(sum(g.square().sum() for g in gradients))),
        norm64=float(torch.sqrt(sum(g.double().square().sum() for g in gradients))))


def main():
    from omegaconf import OmegaConf,open_dict
    from run_aldp_chart_campaign import sha256,verify_source,write_json,receipt
    from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
    from tr_asbs.experiment.inner_observation import restore_inner_snapshot
    from tr_asbs.utils.gradient_clip import clip_gradients_with_overflow_fallback
    a=argparse.ArgumentParser();a.add_argument('--campaign',type=Path,required=True)
    c=a.parse_args().campaign;p=json.loads((c/'plan.json').read_text())
    source=verify_source();assert source==p['source_manifest_sha256']
    for n,h in p['frozen_inputs'].items():assert sha256(n)==h,n
    assert torch.cuda.is_available(),'GPU required'
    torch.set_float32_matmul_precision('highest')
    out=c/'diagnostic';assert not out.exists();out.mkdir()
    reports=[]
    for candidate in p['diagnostic_cases']:
        cfg=OmegaConf.load(candidate['config_path'])
        with open_dict(cfg):
            cfg.root=str(out/candidate['candidate']);cfg.fabric.loggers.save_dir=cfg.root
            cfg.checkpoint_directory=str(out/'unused_checkpoints');cfg.sample_directory=str(out/'unused_samples')
        module=GammaASBSModule(cfg)
        for snap in candidate['snapshots']:
            payload=torch.load(snap,map_location='cpu',weights_only=False)
            restore_inner_snapshot(module,snap,expected_provenance=payload['provenance'])
            storage=module.adjoint_buffer.storage;total=len(module.adjoint_buffer)
            assert total==512 and cfg.train_batch_size==64
            for start in range(0,total,64):
                batch={k:torch.stack(v[start:start+64]).to(module.fabric.device) for k,v in storage.items()}
                module.controller_optimizer.zero_grad(set_to_none=True)
                losses=module.adjoint_matching_log_variance_step(batch);loss=losses['loss']
                row=dict(candidate=candidate['candidate'],snapshot=snap,start=start,loss=scalar_audit(loss),
                         raw_logweight=scalar_audit(batch['log_Ph_Pref_raw']),gamma=module.gamma)
                if bool(torch.isfinite(loss)):
                    module.fabric.backward(loss);row['gradient']=gradient_audit(module.controller)
                    try:
                        norm=clip_gradients_with_overflow_fallback(module.fabric,module.controller,module.controller_optimizer,cfg.gradient_clip_val)
                        row['clip_status']='accepted';row['clip_return']=None if norm is None else float(norm)
                    except RuntimeError as error:row['clip_status']='rejected';row['clip_error']=str(error)
                else:row['clip_status']='loss_nonfinite'
                # Independently expose the exact LV vector, no changed production reduction.
                with torch.no_grad():
                    traj=batch['data_traj'];n,steps=traj.shape[:2];steps-=1
                    times=module.prev_integrator.timesteps.to(traj.device)
                    x=traj[:,:-1].reshape(n*steps,60);xn=traj[:,1:].reshape(n*steps,60)
                    t=times[:-1].expand(n,steps).reshape(-1);dt=times.diff().expand(n,steps).reshape(-1)
                    new=module.integrator.transition(x,t,dt);old=module.prev_integrator.transition(x,t,dt)
                    ratio=(new.log_prob(xn)-old.log_prob(xn)).reshape(n,steps).sum(-1)
                    lv=ratio-module.gamma*batch['log_Ph_Pref_raw'].reshape(-1)
                    row['ratio']=scalar_audit(ratio);row['LV_variance']=variance_audit(lv)
                reports.append(json_safe(row))
                assert module.global_step==payload['counters']['global_step']
                assert module.potential.aldp_potential.openmm_configurations_evaluated==0
                write_json(out/'diagnostic.json',dict(rows=reports,status='partial',source_sha256=source))
        del module
    report=dict(status='diagnostic_complete',rows=reports,source_sha256=source,plan_sha256=sha256(c/'plan.json'),
        OpenMM=0,optimizer_updates=0,matmul_precision=torch.get_float32_matmul_precision(),
        limitation='Post-phase buffers and weights; original rejected batch/pre-update weights unavailable. Cannot claim exact rejected-step replay or original cause from finite saved-state results.',scientific_status='not_evaluated')
    write_json(out/'diagnostic.json',report)
    receipt(out,dict(stage='saved_state_numerical_diagnostic',seed=0),dict(source_sha256=source,plan_sha256=sha256(c/'plan.json')))
    print('diagnostic_complete=true',flush=True)

if __name__=='__main__':main()
