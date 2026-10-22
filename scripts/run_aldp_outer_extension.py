#!/usr/bin/env python3
"""Bounded exact-resume transition preflight for the registered third outer."""
import argparse,json,math
from pathlib import Path

def validate_registration(r):
    expected={'schema':'aldp_registered_outer_extension_v1','u_epochs':2000,'h_epochs':1000,'steps_per_epoch':100,'seed':0,'total_outer_loops':3,'saved_outer_loop':2,'saved_phase':'h','saved_phase_epoch':1000,'cached_global_step':600000,'fresh_updates':300000,'final_global_step':900000,'final_Adam_u_steps':600000,'final_Adam_h_steps':300000,'OU_rate':1.,'loss':'LV','decision_bank':'A','test':'closed','expected_new_OpenMM':260224,'OpenMM_cap':320000}
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Outside registered outer extension')

def marginal_cost(cfg):
    training=2000*cfg.buffer_samples_per_epoch+math.prod(cfg.num_samples_for_dual_loss)+12*cfg.eval_samples
    return training+1152+2*cfg.eval_samples_final

def configuration(plan,campaign):
    from omegaconf import OmegaConf,open_dict
    cfg=OmegaConf.load(plan['parent_config_path']);out=campaign/'transition_preflight/seed0'
    with open_dict(cfg):
        cfg.name,cfg.root=out.name,str(out);cfg.fabric.loggers.save_dir=str(out)
        cfg.checkpoint_directory=str(out/'checkpoints');cfg.sample_directory=str(out/'samples')
        cfg.resume=False;cfg.max_outer_loops=3
    if marginal_cost(cfg)!=260224:raise ValueError('Registered marginal cost mismatch')
    return cfg,out

class ProbeComplete(Exception):pass

def run(plan,campaign,source):
    import torch,numpy as np,random,jax
    from omegaconf import OmegaConf
    from run_aldp_chart_campaign import sha256,write_json,receipt
    from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
    from tr_asbs.experiment.jax_inner import JaxInnerBackend
    from tr_asbs.experiment.inner_observation import attach_exact_inner_resume,continuation_training_contract
    reg=plan['registration'];validate_registration(reg)
    for path,s in plan['frozen_inputs'].items():
        if sha256(path)!=s:raise RuntimeError('Frozen SHA mismatch '+path)
    if not torch.cuda.is_available() or not any(x.platform=='gpu' for x in jax.devices()):raise RuntimeError('GPU mandatory')
    cfg,out=configuration(plan,campaign)
    if out.exists():raise RuntimeError('Existing output must be preserved')
    out.mkdir(parents=True);OmegaConf.save(cfg,out/'config.yaml',resolve=True)
    module=GammaASBSModule(cfg);backend=JaxInnerBackend(module).attach()
    payload=torch.load(reg['parent_snapshot_path'],map_location='cpu',weights_only=False)
    def equal(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    events=[];restored={}
    def after_restore(m):
        assert (m.global_step,m.current_epoch,m.outer_step,m.phase,m.phase_epoch)==(600000,6000,1,'h',1000)
        for name,state in payload['models'].items():assert equal(state,getattr(m,name).state_dict()),name
        for name,state in payload['optimizers'].items():assert equal(state,getattr(m,name).state_dict()),name
        for name,state in payload['buffers'].items():
            b=getattr(m,name);assert equal(state['metadata'],b.metadata) and equal(state['storage'],dict(b.storage)),name
        assert torch.equal(torch.get_rng_state(),payload['rng']['torch_cpu'])
        assert equal(torch.cuda.get_rng_state_all(),payload['rng']['torch_cuda'])
        assert equal(np.random.get_state(),payload['rng']['numpy']) and equal(random.getstate(),payload['rng']['python'])
        counts={p:int(backend.states[p].count) for p in ('u','h')};assert counts=={'u':400000,'h':200000}
        restored.update(model_optimizer_buffer_RNG_exact=True,Adam=counts)
        events.append('restored_outer2_h_end')
    marker=attach_exact_inner_resume(module,reg['parent_snapshot_path'],expected_provenance=reg['expected_parent_provenance'],expected_training_contract=continuation_training_contract(cfg),on_restored=after_restore)
    original_dual=module.optimize_gamma
    def dual(*a,**k):
        assert module.outer_step==2 and module.phase=='ready' and module.global_step==600000
        events.append('outer3_dual');return original_dual(*a,**k)
    module.optimize_gamma=dual
    populate=module.populate_buffer_adjoint_matching
    def buffer(*a,**k):
        assert events==['restored_outer2_h_end','outer3_dual']
        assert module.outer_step==2 and module.global_step==600000
        assert equal(module.prev_integrator.sde.u_controller.state_dict(),module.controller.state_dict())
        events.append('teacher_copy_then_u_buffer');return populate(*a,**k)
    module.populate_buffer_adjoint_matching=buffer
    step=module.training_step;updates=0
    def checked(batch,task):
        nonlocal updates
        if updates==20:raise ProbeComplete()
        assert events[-1]=='teacher_copy_then_u_buffer' and task=='adjoint_matching'
        assert module.global_step==600000+updates
        if not step(batch,task):raise RuntimeError('Nonfinite transition update')
        updates+=1;return True
    module.training_step=checked
    try:module.fit()
    except ProbeComplete:pass
    else:raise RuntimeError('Probe did not stop at registered bound')
    backend.flush(optimizers=True)
    counts={p:int(backend.states[p].count) for p in ('u','h')}
    assert module.global_step==600020 and counts=={'u':400020,'h':200000} and marker['initial_online_suppressed']
    count=module.potential.aldp_potential.openmm_configurations_evaluated
    if count>plan['preflight_OpenMM_cap']:raise RuntimeError('Probe OpenMM cap exceeded')
    result={'status':'passed','scientific_status':'not_evaluated','source_sha256':source,'plan_sha256':sha256(campaign/'plan.json'),'config_sha256':sha256(out/'config.yaml'),'restored':restored,'transition_events':events,'global_step':module.global_step,'new_updates':updates,'Adam':counts,'resume_marker':marker,'OpenMM':count,'full_extension_submitted':False,'parent_snapshot_sha256':sha256(reg['parent_snapshot_path'])}
    write_json(out/'result.json',result);receipt(out,dict(stage='outer_transition',candidate='normal',seed=0),dict(source_sha256=source,plan_sha256=result['plan_sha256']))
    print('postflight_verified=true',flush=True)

def main():
    from run_aldp_chart_campaign import verify_source
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();plan=json.loads((a.campaign/'plan.json').read_text());source=verify_source()
    if source!=plan['source_manifest_sha256']:raise RuntimeError('Source mismatch')
    run(plan,a.campaign,source)
if __name__=='__main__':main()
