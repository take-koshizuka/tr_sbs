"""Zero-update same-batch diagnosis of the captured nonfinite gradient."""
import argparse,json,traceback
from pathlib import Path
import torch
from run_aldp_first_nonfinite import safe,audit

def main():
    from omegaconf import OmegaConf,open_dict
    from run_aldp_chart_campaign import verify_source,sha256,write_json,receipt
    from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
    a=argparse.ArgumentParser();a.add_argument('--campaign',type=Path,required=True);c=a.parse_args().campaign;p=json.loads((c/'plan.json').read_text());source=verify_source();assert source==p['source_manifest_sha256']
    for n,h in p['frozen_inputs'].items():assert sha256(n)==h
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');out=c/'diagnostic';assert not out.exists();out.mkdir()
    cfg=OmegaConf.load(p['witness_config'])
    with open_dict(cfg):cfg.root=str(out);cfg.fabric.loggers.save_dir=str(out);cfg.checkpoint_directory=str(out/'unused');cfg.sample_directory=str(out/'unused_samples')
    OmegaConf.save(cfg,out/'config.yaml',resolve=True);m=GammaASBSModule(cfg);w=torch.load(p['witness'],map_location=m.fabric.device,weights_only=False)
    for n,s in w['models'].items():getattr(m,n).load_state_dict(s)
    for n,v in w['counters'].items():setattr(m,n,v)
    b={k:v.to(m.fabric.device) for k,v in w['batch'].items()};rows=[]
    for mode in ['native_anomaly','loss_scaled_1e-20','variance_reduction_fp64']:
        m.controller_optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autograd.detect_anomaly(check_nan=True) if mode=='native_anomaly' else torch.enable_grad():
                if mode=='variance_reduction_fp64':
                    traj=b['data_traj'];n,steps=traj.shape[:2];steps-=1;times=m.prev_integrator.timesteps.to(traj.device);x=traj[:,:-1].reshape(n*steps,60);xn=traj[:,1:].reshape(n*steps,60);t=times[:-1].expand(n,steps).reshape(-1);dt=times.diff().expand(n,steps).reshape(-1)
                    ratio=(m.integrator.transition(x,t,dt).log_prob(xn)-m.prev_integrator.transition(x,t,dt).log_prob(xn).detach()).reshape(n,steps).sum(-1)
                    lv=ratio.double()-m.gamma*b['log_Ph_Pref_raw'].reshape(-1).double();loss=lv.var(unbiased=True)
                else:loss=m.adjoint_matching_log_variance_step(b)['loss']
                row=dict(mode=mode,loss=audit(loss));(loss*1e-20 if mode=='loss_scaled_1e-20' else loss).backward()
            row['status']='backward_returned'
        except Exception as e:row=dict(mode=mode,status='backward_error',error=str(e),traceback=traceback.format_exc())
        row['gradients']={n:audit(v.grad) for n,v in m.controller.named_parameters() if v.grad is not None};rows.append(safe(row))
    assert m.potential.aldp_potential.openmm_configurations_evaluated==0
    write_json(out/'backward.json',dict(rows=rows,OpenMM=0,optimizer_updates=0,witness_reason=w['reason'],witness_counters=w['counters'],source_sha256=source,plan_sha256=sha256(c/'plan.json'),limits='same captured batch/weights; scale and FP64 reduction are diagnostic only; production unchanged'))
    receipt(out,dict(stage='witness_backward_diagnostic',seed=0),dict(source_sha256=source,plan_sha256=sha256(c/'plan.json')));print('backward_diagnostic_complete=true',flush=True)
if __name__=='__main__':main()
