"""Capture the first rejected shared Torch update, without changing its mathematics."""
import argparse,json,math
from pathlib import Path
import torch

def safe(x):
    if isinstance(x,float) and not math.isfinite(x): return 'nan' if math.isnan(x) else ('+inf' if x>0 else '-inf')
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [safe(v) for v in x]
    return x

def audit(t):
    t=t.detach();return dict(shape=list(t.shape),nonfinite=int((~torch.isfinite(t)).sum()),max_abs=float(t.abs().max()))

class Captured(Exception):pass
class LimitReached(Exception):pass

def install(module,out,plan):
    from tr_asbs.experiment.inner_observation import _rng_state
    import tr_asbs.experiment.train_gamma_asbs_iw as shared
    from run_aldp_chart_campaign import write_json
    original=module.training_step;lv=module.adjoint_matching_log_variance_step;clip=shared.clip_gradients_with_overflow_fallback
    context={}
    def save(reason,error=None):
        # No optimizer update occurred: original returns/rejects before optimizer.step.
        payload=dict(schema='first_nonfinite_preupdate_v1',reason=reason,batch=context['batch'],
            pre_forward_rng=context['rng'],failure_rng=_rng_state(module),sampling_metadata=module.sampling_metadata(),
            models={n:getattr(module,n).state_dict() for n in ['controller','corrector','prev_integrator','prev_corrector']},
            optimizers={n:getattr(module,n).state_dict() for n in ['controller_optimizer','corrector_optimizer']},
            buffers={n:dict(storage=dict(getattr(module,n).storage),metadata=getattr(module,n).metadata,max_size=getattr(module,n).max_size) for n in ['adjoint_buffer','corrector_buffer']},
            counters={n:getattr(module,n) for n in ['current_epoch','global_step','outer_step','phase','phase_epoch','gamma','V0','gamma_h'] if hasattr(module,n)},
            gradients={n:p.grad.detach().clone() for n,p in module.controller.named_parameters() if p.grad is not None},
            loss=context.get('loss'),plan=plan)
        torch.save(payload,out/'witness.pt')
        grads=payload['gradients'];summary=dict(reason=reason,error=error,epoch=module.current_epoch,step=module.global_step,
            batch={k:audit(v) for k,v in context['batch'].items()},loss=None if context.get('loss') is None else audit(context['loss']),
            gradient_nonfinite=sum(int((~torch.isfinite(g)).sum()) for g in grads.values()),
            gradient_norm64=float(torch.sqrt(sum(g.double().square().sum() for g in grads.values()))) if grads else None,
            OpenMM=int(module.potential.aldp_potential.openmm_configurations_evaluated),optimizer_update_at_witness=False)
        write_json(out/'witness.json',safe(summary));raise Captured(reason)
    def loss(batch):
        losses=lv(batch);context['loss']=losses['loss'].detach().clone()
        if not torch.isfinite(losses['loss']):save('loss_nonfinite')
        return losses
    def clipping(*args,**kwargs):
        try:return clip(*args,**kwargs)
        except RuntimeError as e:
            if 'non-finite' in str(e):save('gradient_or_norm_nonfinite',str(e))
            raise
    def step(batch,task):
        if task!='adjoint_matching' or module.current_epoch>=plan['stop_epoch']:raise LimitReached('Registered prefix limit')
        if module.potential.aldp_potential.openmm_configurations_evaluated>plan['OpenMM_cap']:raise LimitReached('OpenMM cap')
        context.clear();context.update(batch=batch,rng=_rng_state(module))
        previous_clip=shared.clip_gradients_with_overflow_fallback
        shared.clip_gradients_with_overflow_fallback=clipping
        try:
            result=original(batch,task)
            if not result:save('unexpected_reject')
            return result
        finally:
            shared.clip_gradients_with_overflow_fallback=previous_clip
    module.adjoint_matching_log_variance_step=loss;module.training_step=step

def main():
    from run_aldp_chart_campaign import verify_source,sha256,train_case,write_json,receipt
    from run_aldp_noise_solver_grid import configuration,GridObserver
    from tr_asbs.process.chart_source import load_calibration_bundle
    a=argparse.ArgumentParser();a.add_argument('--campaign',type=Path,required=True);c=a.parse_args().campaign;p=json.loads((c/'plan.json').read_text());source=verify_source();assert source==p['source_manifest_sha256']
    for n,h in p['frozen_inputs'].items():assert sha256(n)==h
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    cfg,out=configuration(p,c,'full',p['candidate']);assert not out.exists();observer=GridObserver(load_calibration_bundle(cfg.potential.calibration_bundle_path),out,cfg,p,source,'full',p['candidate'])
    try:
        train_case(dict(p,max_openmm_per_geometry_run=p['OpenMM_cap']),c,'full','low_density',0,source,config_and_output=(cfg,out),observer=observer,module_setup=lambda m:install(m,out,p))
        raise RuntimeError('Unexpected full training completion')
    except Captured:
        receipt(out,dict(stage='first_nonfinite_diagnostic',seed=0),dict(source_sha256=source,plan_sha256=sha256(c/'plan.json')));print('witness_captured=true',flush=True)
    except LimitReached as e:
        write_json(out/'limit.json',dict(status='not_reproduced_within_registered_prefix',reason=str(e)));print('prefix_limit_without_witness=true',flush=True)
if __name__=='__main__':main()
