#!/usr/bin/env python3
"""Registered one-axis/paired continuation through shared physical ALDP fit.

Plans are prepared only after an audited normal curve. This script has no Slurm
submission command. Cached prefix costs and newly executed costs stay separate.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

import torch

from analyze_aldp_inner_curve import prefix_checks, screen, verified_inputs
from run_aldp_chart_campaign import ROOT, receipt, sha256, train_case, verify_source, write_json
from run_aldp_jax_inner_curve import InnerCurveObserver
from run_aldp_stability_campaign import StabilityObserver
from tr_asbs.experiment.inner_observation import (
    attach_exact_inner_resume, continuation_training_contract, observe_inner_boundaries,
    save_inner_snapshot,
)


def policy_checks(candidate, reference, limits):
    """Both losses use the identical normal-outer1 held-out BC batch."""
    return prefix_checks({'validation':candidate['matched_h_validation'],'policy':candidate['policy']},
        {'validation':reference['matched_h_validation'],'policy':reference['policy']},limits)


def make_one_axis_cases(report, screening, run):
    rows = report['diagnostics']['inner_boundaries']
    def boundary(phase, epoch):
        return next(row for row in rows if row['phase']==phase and row['phase_epoch']==epoch)
    cases={}
    for name,phase,epoch,u,h in (
        ('normal','h',1000,2000,1000),
        ('u_short','u',screening['phases']['u']['proposed_shorter_epoch'],
         screening['phases']['u']['proposed_shorter_epoch'],1000),
        ('h_short','h',screening['phases']['h']['proposed_shorter_epoch'],
         2000,screening['phases']['h']['proposed_shorter_epoch'])):
        if epoch is None:
            continue
        point=boundary(phase,epoch)
        cases[name]=dict(u_epochs=u,h_epochs=h,outer_loops=2,seed=0,
            checkpoint_path=str(run/point['snapshot']),saved_phase=phase,saved_phase_epoch=epoch,
            cached_global_step=100*(epoch+(2000 if phase=='h' else 0)),
            expected_provenance=report['provenance'],parent_boundary=point,
            reuse_first_outer_policy=phase=='h',
        first_h_validation_path=str(run/'heldout_h.pt') if phase=='h' else None,
            save_first_h_prefix=screening['phases']['h']['proposed_shorter_epoch'] if name=='u_short' else None)
    if len(cases)==1:
        return {}  # No shorter proposal: an isolated new control is unnecessary.
    return cases


def verify_plan(plan):
    if plan.get('schema') != 'aldp_inner_continuation_plan_v1':
        raise ValueError('Registered continuation plan required')
    if plan['study_stage'] not in ('one_axis', 'paired'):
        raise ValueError('Unknown continuation stage')
    allowed = {'normal', 'u_short', 'h_short'} if plan['study_stage']=='one_axis' else {'paired'}
    if not plan['cases'] or not set(plan['cases']) <= allowed:
        raise ValueError('Unregistered continuation case')
    contract=plan['curve_contract']
    if contract['selection_bank']!='a' or contract['test_bank']!='closed':
        raise ValueError('Keep A-only selection and test closed')
    for name,case in plan['cases'].items():
        u,h=case['u_epochs'],case['h_epochs']
        if (u not in contract['milestones']['u'] or h not in contract['milestones']['h']
            or case['outer_loops']!=2 or case['seed']!=0):
            raise ValueError('Continuation lies outside registered grid/outer/seed')
        if (name=='normal' and (u,h)!=(2000,1000)
            or name=='u_short' and (u>=2000 or h!=1000 or case['saved_phase']!='u')
            or name=='h_short' and (u!=2000 or h>=1000 or case['saved_phase']!='h')
            or name=='paired' and (u>=2000 or h>=1000 or case['saved_phase']!='h')):
            raise ValueError('Case changed more than its registered phase durations')
        expected=100*(case['saved_phase_epoch']+(u if case['saved_phase']=='h' else 0))
        if case['cached_global_step']!=expected:
            raise ValueError('Cached prefix belongs to a different u/h history')


def expected_cost(cfg, case):
    """Marginal work after the exact registered first-outer prefix."""
    u,h=case['u_epochs'],case['h_epochs']
    start=case['cached_global_step']//cfg.steps_per_epoch
    end=2*(u+h)
    online=end//cfg.log_detail_interval_epoch-start//cfg.log_detail_interval_epoch
    training=u*cfg.buffer_samples_per_epoch+cfg.num_samples_for_dual_loss[0]*cfg.num_samples_for_dual_loss[1]+online*cfg.eval_samples
    phases=3 if case['saved_phase']=='u' else 2
    first_probe=0 if case['reuse_first_outer_policy'] else 2*cfg.eval_samples_final+128
    diagnostic=phases*(2*128+128)+first_probe+256+128
    final=2*cfg.eval_samples_final
    return dict(training_and_online_openmm_configurations=training,
        diagnostic_openmm_configurations=diagnostic,final_evaluation_openmm_configurations=final,
        total_openmm_configurations=training+diagnostic+final,
        new_optimizer_steps=cfg.steps_per_epoch*end-case['cached_global_step'],
        final_optimizer_steps=cfg.steps_per_epoch*end,
        u_adam_steps=2*u*cfg.steps_per_epoch,h_adam_steps=2*h*cfg.steps_per_epoch)


def configuration(plan, campaign, candidate):
    from omegaconf import OmegaConf,open_dict
    case=plan['cases'][candidate]
    cfg=OmegaConf.load(plan['normal_config_path'])
    output=campaign/'continuation'/candidate
    if case['outer_loops'] != 2 or case['seed'] != 0:
        raise ValueError('Continuation uses only two total outers and the registered seed0')
    with open_dict(cfg):
        cfg.name,cfg.root=output.name,str(output)
        cfg.fabric.loggers.save_dir=str(output)
        cfg.checkpoint_directory=str(output/'checkpoints')
        cfg.sample_directory=str(output/'samples')
        cfg.adjoint_matching_epochs, cfg.corrector_matching_epochs=case['u_epochs'],case['h_epochs']
        cfg.initial_adjoint_matching_epochs=None
        cfg.max_outer_loops=2
        cfg.resume=False  # Exact adapter activates resume after attaching JAX.
        cfg.exact_inner_resume_input=dict(path=case['checkpoint_path'],
            provenance=case['expected_provenance'],legacy_resume_used=False)
    if continuation_training_contract(cfg) != plan['training_contract']:
        raise ValueError('Continuation changed fixed inner-training settings')
    return cfg,output


class ContinuationObserver(InnerCurveObserver):
    def __init__(self,bundle,output,cfg,plan,case,source_sha):
        StabilityObserver.__init__(self,bundle,output,1024,128,cfg.potential.chart_key)
        self.cfg,self.output,self.case=cfg,output,case
        self.contract=dict(quality_samples=1024,validation_samples=256)
        self.provenance=dict(source_sha256=source_sha,plan_sha256=sha256(output.parent.parent/'plan.json'),
            config_sha256=sha256(output/'config.yaml'),calibration_sha256=sha256(cfg.potential.calibration_bundle_path),
            parent_snapshot_sha256=sha256(case['checkpoint_path']))
        self.validation={};self.started=time.perf_counter()
        self.comparison_h_validation_path=plan['comparison_h_validation_path']
        self.report.update(inner_boundaries=[],reused_prefix_boundaries=[],
            cached_global_step=case['cached_global_step'],parent_snapshot=case['checkpoint_path'],
            cached_reference_paid_openmm=plan['cached_reference_paid_openmm'],
            minimum_inner_duration='not_established',bc_losses_across_different_u_are_not_matched=True)

    def _matched_h_loss(self,module):
        # Keep the current-u BC batch for within-phase curves, and report a
        # separate shared-bank risk for comparisons across different u models.
        own=self.validation.get('h')
        try:
            self.validation['h']=self.comparison_h_validation
            values=self._validation_loss(module,'h')
            values['bank']='identical normal-outer1 frozen endpoints/bridge draws'
            values['batch_path']=self.comparison_h_validation_path
            return values
        finally:
            if own is None:self.validation.pop('h',None)
            else:self.validation['h']=own

    def _validation_start(self,module,phase):
        super()._validation_start(module,phase)
        (self.output/f'heldout_{phase}.pt').replace(
            self.output/f'heldout_{phase}_outer{module.outer_step+1:03d}.pt')

    @torch.no_grad()
    def __call__(self,module,event,payload=None):
        from tr_asbs.experiment.inner_observation import isolated_inner_diagnostic
        with isolated_inner_diagnostic(module):
            if event=='initial':
                self._instrument(module)
                self.report['training_contract']=dict(u_epochs=module.adjoint_matching_epochs,
                    h_epochs=module.corrector_matching_epochs,steps_per_epoch=module.cfg.steps_per_epoch,outer_loops=2)
                return
            if event=='restored':
                torch.manual_seed(12455)
                self.probe=module.source.sample((64,)).detach()
                self.report['restored_counts']=dict(epoch=module.current_epoch,global_step=module.global_step,
                    phase=module.phase,phase_epoch=module.phase_epoch,outer_step=module.outer_step)
                data=torch.load(self.comparison_h_validation_path,map_location='cpu',weights_only=False)
                self.comparison_h_validation={key:value.to(module.fabric.device) for key,value in data.items()}
                if self.case['saved_phase']=='h':
                    data=torch.load(self.case['first_h_validation_path'],map_location='cpu',weights_only=False)
                    self.validation['h']={key:value.to(module.fabric.device) for key,value in data.items()}
                    self.report['first_outer_complete']=(deepcopy(self.case['parent_boundary']['policy'])
                        if self.case['reuse_first_outer_policy'] else self.policy(module,'first_outer_complete'))
                    self.report['first_outer_own_h_validation']=self._validation_loss(module,'h')
                    self.report['first_outer_validation']=self._matched_h_loss(module)
                return
            if event=='phase_end' and module.global_step==self.case['cached_global_step']:
                self.report['reused_prefix_boundaries'].append(payload)
                return
            if event=='inner_boundary':
                phase,outer=payload['phase'],payload['outer_loop']
                if payload['phase_epoch']==0:
                    self._validation_start(module,phase)
                label=f"outer{outer:03d}_{phase}_{payload['phase_epoch']:04d}"
                path=save_inner_snapshot(module,self.output/'inner_snapshots'/f'{label}.pt',self.provenance)
                validation=self._validation_loss(module,phase) if phase in self.validation else None
                self.report['inner_boundaries'].append(dict(payload,snapshot=str(path.relative_to(self.output)),
                    validation=validation,openmm_configurations=module.potential.aldp_potential.openmm_configurations_evaluated))
                write_json(self.output/'continuation_progress.json',dict(status='running',
                    boundaries=self.report['inner_boundaries'],minimum='not_established'))
                return
            if event=='final':
                super().__call__(module,event,payload)
                from tr_asbs.utils.eval_logweight import compute_raw_path_ess
                z,_=module.potential.chart(payload['y'])
                indices=self.bundle['phi_psi_native_indices']
                delta=(z[:,indices]*module.potential.chart.scales[indices]
                    -module.potential.chart.offsets[indices]).cpu().numpy()
                import numpy as np
                self.report['trained_policy'].update(
                    raw_path_ess_fraction=float(compute_raw_path_ess(payload['log_Ph_Pref'])),
                    raw_log_path_ratio_variance=float(payload['log_Ph_Pref'].double().var(unbiased=False)),
                    achieved_forward_kl=float(payload['log_Pref_Pu_det'].double().mean()),
                    near_cut_fraction_phi_psi=np.mean(np.pi-np.abs(delta)<.15,axis=0).tolist())
                self.report['next_outer_validation']=self._matched_h_loss(module)
                self.report['resume_marker']=module.exact_inner_resume
                return self.report
            result=StabilityObserver.__call__(self,module,event,payload)
            if event=='phase_end' and payload['outer_loop']==1 and payload['phase']=='h':
                self.report['first_outer_complete']=self.policy(module,'first_outer_complete')
                self.report['first_outer_own_h_validation']=self._validation_loss(module,'h')
                self.report['first_outer_validation']=self._matched_h_loss(module)
            return result


def setup_backend(module,observer,plan):
    from tr_asbs.experiment.jax_inner import JaxInnerBackend
    JaxInnerBackend(module).attach()
    original=module.training_step
    def step(batch,task):
        success=original(batch,task)
        if not success:
            write_json(observer.output/'numerical_failure.json',dict(task=task,
                epoch=module.current_epoch,global_step=module.global_step,
                openmm_configurations=module.potential.aldp_potential.openmm_configurations_evaluated))
            raise RuntimeError('Nonfinite continuation update; preserve last complete boundary')
        return success
    module.training_step=step
    keep=observer.case['save_first_h_prefix']
    observe_inner_boundaries(module,lambda data:module.campaign_observe('inner_boundary',data),
        {'u':[],'h':[keep] if keep else []})
    attach_exact_inner_resume(module,observer.case['checkpoint_path'],
        expected_provenance=observer.case['expected_provenance'],
        expected_training_contract=plan['training_contract'],
        on_restored=lambda m:m.campaign_observe('restored'))


def run_case(plan,campaign,candidate,source_sha):
    from omegaconf import OmegaConf
    from tr_asbs.process.chart_source import load_calibration_bundle
    verify_plan(plan)
    cfg,out=configuration(plan,campaign,candidate);case=plan['cases'][candidate]
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Preserve previous attempt; use a new source/output for retries')
    out.mkdir(parents=True);OmegaConf.save(cfg,out/'config.yaml',resolve=True)
    observer=ContinuationObserver(load_calibration_bundle(cfg.potential.calibration_bundle_path),out,cfg,plan,case,source_sha)
    cost=expected_cost(cfg,case)
    train_case(dict(plan,max_openmm_per_geometry_run=plan['openmm_cap']),campaign,'continuation',
        cfg.potential.chart_key,0,source_sha,config_and_output=(cfg,out),observer=observer,
        provenance=observer.provenance,module_setup=lambda module:setup_backend(module,observer,plan))
    report=json.loads((out/'result.json').read_text())
    if report['global_step'] != cost['final_optimizer_steps']:
        raise RuntimeError('Continuation did not complete the registered two-outer update count')
    for key,value in cost.items():
        if key.endswith('configurations') and report['cost'][key] != value:
            raise RuntimeError('Marginal continuation cost mismatch: '+key)
    counts=report['diagnostics']['backend_adam_update_counts']
    if counts != {'u':cost['u_adam_steps'],'h':cost['h_adam_steps']}:
        raise RuntimeError('Cached plus new Adam update counts differ from the registered durations')
    if not report['diagnostics']['resume_marker']['initial_online_suppressed']:
        raise RuntimeError('Continuation consumed an extra initial online RNG draw')
    report.update(candidate=candidate,minimum_inner_duration='not_established',
        geometry_admission='not_evaluated',new_optimizer_steps=cost['new_optimizer_steps'],
        cached_global_step=case['cached_global_step'],expected_marginal_cost=cost,
        cached_reference_paid_openmm=plan['cached_reference_paid_openmm'])
    write_json(out/'result.json',report)
    receipt(out,dict(stage='continuation',candidate=candidate,seed=0),
        dict(source_sha256=source_sha,provenance=observer.provenance))
    return out


def prepare_one_axis(normal_campaign,audit,campaign,source_sha):
    from omegaconf import OmegaConf
    run=normal_campaign/'normal_inner_curve/seed0'
    report,parent=verified_inputs(run,normal_campaign/'plan.json',audit)
    proposal=screen(report,parent['curve_contract'])
    cases=make_one_axis_cases(report,proposal,run)
    if not cases:
        raise ValueError('No shorter prefixes passed; do not submit an unnecessary new control')
    if (campaign/'plan.json').exists():
        raise RuntimeError('Prepared continuation plan exists; never replace its frozen inputs')
    rec=json.loads((run/'receipt.json').read_text())
    frozen=dict(parent['frozen_inputs'])
    for path in (normal_campaign/'plan.json',audit,run/'receipt.json'):
        frozen[str(path)]=sha256(path)
    for name,digest in rec['files'].items():
        frozen[str(run/name)]=digest
    # Freeze an audit copy: scheduled monitor_state.json continues to change.
    audit_copy=campaign/'normal_completion_audit.json'
    if audit_copy.exists():raise RuntimeError('Existing audit copy; reconcile preparation before retry')
    audit_copy.parent.mkdir(parents=True,exist_ok=True)
    audit_copy.write_bytes(audit.read_bytes())
    frozen.pop(str(audit),None);frozen[str(audit_copy)]=sha256(audit_copy)
    plan=dict(schema='aldp_inner_continuation_plan_v1',study_stage='one_axis',cases=cases,
        source_manifest_sha256=source_sha,frozen_inputs=frozen,
        normal_config_path=str(run/'config.yaml'),normal_report_path=str(run/'result.json'),
        comparison_h_validation_path=str(run/'heldout_h.pt'),
        backend_verification_path=parent['backend_verification_path'],
        backend_job_id=parent['backend_job_id'],jax_backend_sha256=parent['jax_backend_sha256'],
        curve_contract=parent['curve_contract'],training_contract=continuation_training_contract(OmegaConf.load(run/'config.yaml')),
        cached_reference_paid_openmm=report['cost']['total_openmm_configurations'],
        openmm_cap=320000,quality='not_evaluated',minimum='not_established')
    verify_plan(plan)
    write_json(campaign/'plan.json',plan);write_json(campaign/'prefix_screening.json',proposal)
    return plan


def compare_one_axis(reports,plan):
    """Conditional seed0 decisions after receipt/accounting gates in the monitor."""
    if set(reports)!=set(plan['cases']) or 'normal' not in reports:
        raise ValueError('All registered cases, including the cached-prefix normal control, required')
    limits=plan['curve_contract']['non_deterioration_limits']
    normal=reports['normal']['diagnostics']
    result=dict(schema='aldp_inner_continuation_comparison_v1',decision_bank='a',disclosure_bank='b',
        test='closed',minimum_inner_duration='not_established',geometry_admission='not_evaluated',
        scope='registered finite grid, seed0, two total outers; conditional adequacy only',candidates={})
    for name,report in reports.items():
        diagnostic=report['diagnostics'];points={}
        for key,loss in (('first_outer_complete','first_outer_validation'),('trained_policy','next_outer_validation')):
            checks=policy_checks({'policy':diagnostic[key],'matched_h_validation':diagnostic[loss]},
                {'policy':normal[key],'matched_h_validation':normal[loss]},limits)
            points[key]=dict(checks=checks,passed=all(row['passed'] for row in checks.values()),
                raw_physical_energy_tail_quantiles=diagnostic[key]['physical_energy']['quantiles'],
                disclosure_b=diagnostic[key]['metrics']['b'])
        result['candidates'][name]=dict(conditional_gates_passed=all(p['passed'] for p in points.values()),
            points=points,cost=report['cost'],new_optimizer_steps=report['new_optimizer_steps'],
            cached_global_step=report['cached_global_step'],cached_reference_paid_openmm=report['cached_reference_paid_openmm'])
    passed=[name for name,row in result['candidates'].items() if name!='normal' and row['conditional_gates_passed']]
    result['passing_shorter_cases']=passed
    result['paid_study_openmm_configurations']=plan.get('cached_reference_paid_openmm',0)+sum(
        r['cost']['total_openmm_configurations'] for r in reports.values())
    result['pair_status']='not_requested_one_axis_failed_or_not_proposed'
    if set(passed)=={'u_short','h_short'}:
        udiag=reports['u_short']['diagnostics'];h=plan['cases']['h_short']['h_epochs']
        point=next(row for row in udiag['inner_boundaries']
            if row['outer_loop']==1 and row['phase']=='h' and row['phase_epoch']==h)
        loss,full=point['validation']['heldout_loss'],udiag['first_outer_own_h_validation']['heldout_loss']
        import math
        within=math.isfinite(loss) and math.isfinite(full) and loss<=limits['heldout_loss_ratio']*full
        result['paired_prefix_check']=dict(heldout_loss=loss,bound=limits['heldout_loss_ratio']*full,
            passed=within,bank='same shortened-u endpoints and bridge draws',snapshot=point['snapshot'])
        result['pair_status']='ready_for_one_paired_confirmation' if within else 'heldout_bc_under_shortened_u_failed'
    return result


def prepare_paired(one_axis,audit,campaign,source_sha):
    """Reuse the shortened-u h-prefix; never take h from the normal-u run."""
    parent=json.loads((one_axis/'plan.json').read_text());verify_plan(parent)
    proof=json.loads(audit.read_text())
    if (proof.get('status')!='continuations_verified' or proof['source_sha256']!=source_sha
        or proof['plan_sha256']!=sha256(one_axis/'plan.json') or parent['study_stage']!='one_axis'):
        raise ValueError('Verified one-axis continuation/accounting audit required')
    reports={};frozen=dict(parent['frozen_inputs'])
    for name in parent['cases']:
        out=one_axis/'continuation'/name;rec=json.loads((out/'receipt.json').read_text())
        for filename,digest in rec['files'].items():
            path=Path(filename)
            if path.is_absolute() or '..' in path.parts or sha256(out/path)!=digest:
                raise ValueError('Continuation receipt path/SHA mismatch')
            frozen[str(out/path)]=digest
        if proof['results_sha256'][name]!=sha256(out/'result.json'):
            raise ValueError('One-axis result audit SHA mismatch')
        frozen[str(out/'receipt.json')]=sha256(out/'receipt.json')
        reports[name]=json.loads((out/'result.json').read_text())
    decision=compare_one_axis(reports,parent)
    if decision['pair_status']!='ready_for_one_paired_confirmation':
        raise ValueError('Both axes and the coupled BC prefix must pass before a pair')
    out=one_axis/'continuation/u_short';report=reports['u_short']
    u,h=parent['cases']['u_short']['u_epochs'],parent['cases']['h_short']['h_epochs']
    point=next(row for row in report['diagnostics']['inner_boundaries']
        if row['outer_loop']==1 and row['phase']=='h' and row['phase_epoch']==h)
    case=dict(u_epochs=u,h_epochs=h,outer_loops=2,seed=0,checkpoint_path=str(out/point['snapshot']),
        saved_phase='h',saved_phase_epoch=h,cached_global_step=100*(u+h),
        expected_provenance=report['provenance'],parent_boundary=point,
        reuse_first_outer_policy=False,first_h_validation_path=str(out/'heldout_h_outer001.pt'),
        save_first_h_prefix=None)
    if campaign.exists() and any(campaign.iterdir()):
        raise RuntimeError('Paired campaign exists; preserve and reconcile before retry')
    campaign.mkdir(parents=True,exist_ok=True)
    copy=campaign/'one_axis_completion_audit.json';copy.write_bytes(audit.read_bytes())
    frozen[str(copy)]=sha256(copy);frozen[str(one_axis/'plan.json')]=sha256(one_axis/'plan.json')
    plan=dict(parent,study_stage='paired',cases={'paired':case},frozen_inputs=frozen,
        one_axis_campaign=str(one_axis),one_axis_decision=decision,source_manifest_sha256=source_sha,
        cached_upstream_completed_openmm=decision['paid_study_openmm_configurations'])
    verify_plan(plan);write_json(campaign/'plan.json',plan)
    return plan


def compare_paired(report,plan):
    parent=Path(plan['one_axis_campaign'])
    normal=json.loads((parent/'continuation/normal/result.json').read_text())
    comparison=compare_one_axis({'normal':normal,'paired':report},dict(plan,cases={'normal':{},'paired':{}}))
    comparison['pair_status']='paired_confirmation_passed' if comparison['candidates']['paired']['conditional_gates_passed'] else 'paired_confirmation_failed'
    comparison['paid_study_openmm_configurations']=plan['cached_upstream_completed_openmm']+report['cost']['total_openmm_configurations']
    comparison['cached_upstream_completed_openmm']=plan['cached_upstream_completed_openmm']
    comparison['recommendation_role']='conditional finite-grid/seed0 choice; physical admission and convergence remain unachieved'
    comparison['one_axis_decision']=plan['one_axis_decision']
    return comparison


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',required=True,type=Path)
    parser.add_argument('--candidate')
    parser.add_argument('--prepare-one-axis',action='store_true')
    parser.add_argument('--prepare-paired',action='store_true')
    parser.add_argument('--normal-campaign',type=Path)
    parser.add_argument('--one-axis-campaign',type=Path)
    parser.add_argument('--audit',type=Path)
    args=parser.parse_args();source=verify_source()
    if args.prepare_one_axis and args.prepare_paired:parser.error('Choose one preparation stage')
    if args.prepare_one_axis:
        if args.normal_campaign is None or args.audit is None:
            parser.error('Plan preparation requires normal campaign and completed audit')
        plan=prepare_one_axis(args.normal_campaign.resolve(),args.audit.resolve(),args.campaign.resolve(),source)
        print(json.dumps(dict(status='prepared_no_jobs',cases=list(plan['cases']))));return
    if args.prepare_paired:
        if args.one_axis_campaign is None or args.audit is None:
            parser.error('Paired preparation requires audited one-axis campaign')
        plan=prepare_paired(args.one_axis_campaign.resolve(),args.audit.resolve(),args.campaign.resolve(),source)
        print(json.dumps(dict(status='prepared_no_jobs',cases=list(plan['cases']))));return
    plan=json.loads((args.campaign/'plan.json').read_text())
    verify_plan(plan)
    if source != plan['source_manifest_sha256']:
        raise RuntimeError('Continuation source changed')
    for name,digest in plan['frozen_inputs'].items():
        if sha256(name) != digest:raise RuntimeError('Frozen continuation input SHA mismatch: '+name)
    if args.candidate not in plan['cases']:
        parser.error('Choose exactly one registered continuation case')
    if not torch.cuda.is_available():raise RuntimeError('Use the configured remote GPU')
    import jax
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    try:
        run_case(plan,args.campaign.resolve(),args.candidate,source)
        print('postflight_verified=true',flush=True)
    except Exception as error:
        write_json(args.campaign/'continuation'/args.candidate/'failure.json',dict(status='failed',
            error=repr(error),traceback=traceback.format_exc(),scientific_status='not_evaluated',source_sha256=source))
        raise


if __name__=='__main__':main()
