#!/usr/bin/env python3
"""Registered noise/reference x solver grid through the shared Torch trainer."""
import argparse
import json
from pathlib import Path
import traceback

import torch
from run_aldp_chart_campaign import ROOT, receipt, sha256, train_case, verify_source, write_json
from run_aldp_component_campaign import kernel_contract
from run_aldp_jax_inner_curve import InnerCurveObserver

PRESETS = ('brownian_constant', 'brownian_linear', 'brownian_geometric',
           'brownian_cosine', 'brownian_edm', 'ou', 'vp')
SOLVERS = ('euler', 'legacy', 'diffusion_euler', 'linear_exponential', 'linear_exponential_midpoint')


def incompatibility(preset, solver, shifted_chart_source=False):
    if preset not in PRESETS or solver not in SOLVERS:
        raise ValueError('Unknown grid axis')
    if preset.startswith('brownian_') and solver.startswith('linear_exponential'):
        return 'LinearExponentialIntegrator requires LinearGaussianReference'
    if preset in ('ou', 'vp') and solver in ('legacy', 'diffusion_euler'):
        return 'This integrator requires ZeroDriftSDE; OU/VP have nonzero drift'
    if preset == 'vp' and shifted_chart_source:
        return 'VP requires stationary Gaussian source; shifted ChartMappedGaussianPrior is non-Gaussian'
    return None


def matrix(shifted_chart_source=False):
    return [dict(candidate=p+'__'+s, preset=p, solver=s,
                 incompatible_reason=incompatibility(p, s, shifted_chart_source)) for p in PRESETS for s in SOLVERS]


def configuration(plan, campaign, stage, candidate):
    from omegaconf import OmegaConf, open_dict
    from tr_asbs.experiment.reference_config import configure_reference
    cell = next(c for c in plan['matrix'] if c['candidate'] == candidate)
    if incompatibility(cell['preset'], cell['solver']):
        raise ValueError('Incompatible cell cannot be submitted')
    if stage not in ('smoke', 'full'):
        raise ValueError('Unknown stage')
    cfg = OmegaConf.load(ROOT / plan['base_config_relative'])
    output = campaign / stage / candidate
    if cell['incompatible_reason']:
        raise ValueError('Source-incompatible cell cannot be submitted')
    if cell['preset'] == 'vp' and str(cfg.source.get('_target_', '')).endswith('ChartMappedGaussianPrior') and cfg.source.chart_key != 'current':
        raise ValueError('VP requires stationary Gaussian source; shifted ChartMappedGaussianPrior is non-Gaussian')
    smoke = stage == 'smoke'
    with open_dict(cfg):
        cfg.root, cfg.name = str(output), candidate
        cfg.fabric.loggers.save_dir = str(output)
        cfg.checkpoint_directory = str(output/'checkpoints')
        cfg.sample_directory = str(output/'samples')
        cfg.seed = 0
        cfg.reference_preset = cell['preset']
        cfg.reference_solver = 'linear_exponential' if cell['solver'].startswith('linear_exponential') else cell['solver']
        cfg.reference_variance = plan['integrated_diffusion_variance']
        cfg.reference_g_min, cfg.reference_g_max = .1, 1.
        cfg.reference_steps, cfg.reference_noise_knots = 200, 200
        cfg.reference_grid = 'uniform_t'
        cfg.reference_control_evaluation = 'midpoint' if cell['solver'].endswith('_midpoint') else 'left'
        cfg.max_outer_loops = 1 if smoke else plan['outer_loops']
        cfg.adjoint_matching_epochs = 1 if smoke else 2000
        cfg.corrector_matching_epochs = 1 if smoke else 1000
        cfg.steps_per_epoch = 2 if smoke else 100
        cfg.initial_adjoint_matching_epochs = None
        cfg.eval_samples_final = 8 if smoke else 1024
        cfg.eval_samples = 8 if smoke else 64
        cfg.log_detail_interval_epoch = 1 if smoke else 250
        # The shared collector batches inference_batch_size paths, so fewer
        # than 64/2=32 groups produce zero iterations, not a small dual bank.
        cfg.num_samples_for_dual_loss = [32, 2] if smoke else [32, 8]
        cfg.save_checkpoint_interval_epoch = 3000
        cfg.resume = False
        if plan.get('controller_backward_mode'):
            if smoke or plan['controller_backward_mode'] != 'loss_scaled_fp64_clip':
                raise ValueError('Registered scaled LV continuation is full-only')
            cfg.fc_backward_mode = plan['controller_backward_mode']
    configure_reference(cfg)
    return cfg, output


def expected_cost(cfg, phase_samples):
    groups, paths = map(int, cfg.num_samples_for_dual_loss)
    batch = int(cfg.inference_batch_size)
    if paths < 1 or batch % paths or groups < batch//paths or groups % (batch//paths):
        raise ValueError('Dual bank must contain a positive whole number of inference batches')
    loops = int(cfg.max_outer_loops)
    u, h, n = int(cfg.adjoint_matching_epochs), int(cfg.corrector_matching_epochs), int(cfg.eval_samples_final)
    online = 1 + loops*(u+h)//int(cfg.log_detail_interval_epoch)
    train = loops*u*int(cfg.buffer_samples_per_epoch) + loops*int(cfg.num_samples_for_dual_loss[0])*int(cfg.num_samples_for_dual_loss[1]) + online*int(cfg.eval_samples)
    diagnostic = 3*n + 2*min(128,n) + 2*loops*(2*phase_samples+min(128,phase_samples)) + min(128,n)
    return dict(optimizer_steps=loops*(u+h)*int(cfg.steps_per_epoch),
                training_and_online_openmm_configurations=train,
                diagnostic_openmm_configurations=diagnostic,
                final_evaluation_openmm_configurations=2*n,
                total_openmm_configurations=train+diagnostic+2*n)


class GridObserver(InnerCurveObserver):
    """Reuse current-policy physical diagnostics without JAX-only milestones."""
    def __init__(self, bundle, output, cfg, plan, source_sha, stage, candidate):
        from run_aldp_stability_campaign import StabilityObserver
        StabilityObserver.__init__(self, bundle, output, cfg.eval_samples_final,
                                   8 if stage == 'smoke' else 128, cfg.potential.chart_key)
        self.output, self.cfg = output, cfg
        self.provenance = dict(source_sha256=source_sha, plan_sha256=sha256(output.parent.parent/'plan.json'),
            calibration_sha256=sha256(cfg.potential.calibration_bundle_path),
            candidate=candidate)
        self.report['grid_contract'] = dict(backend='torch', stage=stage, candidate=candidate,
            selection_bank='a', disclosure_bank='b', test_bank='closed')
        if plan.get('controller_backward_mode'):
            self.provenance['controller_backward_mode'] = plan['controller_backward_mode']
            self.provenance['numerical_gate_sha256'] = plan['diagnostic_release_guard']['gate_sha256']

    @torch.no_grad()
    def __call__(self, module, event, payload=None):
        from run_aldp_stability_campaign import StabilityObserver
        if event == 'initial':
            # train_case writes the resolved config before this callback, but
            # after the observer is constructed. Never hash it in __init__.
            self.provenance['config_sha256'] = sha256(self.output/'config.yaml')
            self.report['grid_contract']['kernel'] = kernel_contract(module.integrator)
        if self.provenance.get('controller_backward_mode'):
            self.report['controller_backward_numerics'] = dict(
                mode=module.fc_backward_mode, statistics=dict(module.fc_backward_statistics),
                diagnostic_gate_sha256=self.provenance['numerical_gate_sha256'])
            if event == 'phase_end':
                self.report.setdefault('controller_backward_phase_statistics', []).append(dict(
                    outer_loop=payload['outer_loop'], phase=payload['phase'],
                    statistics=dict(module.fc_backward_statistics)))
        result = StabilityObserver.__call__(self, module, event, payload)
        if event == 'phase_end' and payload['phase'] == 'h':
            from tr_asbs.experiment.inner_observation import save_inner_snapshot
            save_inner_snapshot(module, self.output/f"outer{payload['outer_loop']:02d}_h_exact.pt", self.provenance)
            # Persist curves incrementally so a later failure keeps prior observations.
            write_json(self.output/'progress.json', self.report)
        return result


def verify(directory, plan, source_sha, stage, candidate):
    directory = Path(directory)
    document = json.loads((directory/'receipt.json').read_text())
    cfg, expected_directory = configuration(plan, directory.parent.parent, stage, candidate)
    if directory != expected_directory or not document.get('postflight_verified'):
        raise RuntimeError('Wrong run directory or receipt status')
    if document['identity'] != dict(stage=stage, chart='low_density', seed=0):
        raise RuntimeError('Receipt identity mismatch')
    for relative, value in document['files'].items():
        if sha256(directory/relative) != value:
            raise RuntimeError('Artifact SHA mismatch: '+relative)
    report = json.loads((directory/'result.json').read_text())
    if (report['source_sha256'] != source_sha or report['provenance']['plan_sha256'] != sha256(directory.parent.parent/'plan.json')
            or report['provenance']['candidate'] != candidate or report['rejected_training_steps']):
        raise RuntimeError('Source/plan/candidate/numerical contract mismatch')
    from omegaconf import OmegaConf
    if OmegaConf.to_container(OmegaConf.load(directory/'config.yaml'), resolve=True) != OmegaConf.to_container(cfg, resolve=True):
        raise RuntimeError('Resolved configuration changed')
    expected = expected_cost(cfg, 8 if stage == 'smoke' else 128)
    if report['global_step'] != expected['optimizer_steps']:
        raise RuntimeError('Incomplete optimizer budget')
    for key, value in expected.items():
        if key != 'optimizer_steps' and report['cost'][key] != value:
            raise RuntimeError('Cost mismatch: '+key)
    phases = report['diagnostics']['phase_end']
    if [(p['outer_loop'],p['phase']) for p in phases] != [(o,p) for o in range(1,cfg.max_outer_loops+1) for p in ('u','h')]:
        raise RuntimeError('Incomplete phase curve')
    if len(report['diagnostics']['dual']) != cfg.max_outer_loops:
        raise RuntimeError('Incomplete dual curve')
    for outer in range(1,cfg.max_outer_loops+1):
        if f'outer{outer:02d}_h_exact.pt' not in document['files']:
            raise RuntimeError('Missing exact snapshot')
    if plan.get('controller_backward_mode'):
        numerics = report['diagnostics']['controller_backward_numerics']
        stats = numerics['statistics']
        expected_u = int(cfg.max_outer_loops)*int(cfg.adjoint_matching_epochs)*int(cfg.steps_per_epoch)
        if (numerics['mode'] != plan['controller_backward_mode'] or
            numerics['diagnostic_gate_sha256'] != plan['diagnostic_release_guard']['gate_sha256'] or
            stats['attempts'] != expected_u or stats['updates'] != expected_u or
            stats['loss_nonfinite'] or stats['gradient_nonfinite'] or
            not 0 < stats['minimum_scale'] <= stats['maximum_scale'] <= 1):
            raise RuntimeError('LV backward numerical/counter contract mismatch')
        curve = report['diagnostics']['controller_backward_phase_statistics']
        if [(r['outer_loop'],r['phase']) for r in curve] != [(o,p) for o in range(1,6) for p in ('u','h')]:
            raise RuntimeError('Missing backward numerical phase evidence')
        for r in curve:
            expected = r['outer_loop']*200000
            if r['statistics']['attempts'] != expected or r['statistics']['updates'] != expected:
                raise RuntimeError('Incomplete per-phase LV backward updates')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--stage', choices=('smoke','full'), required=True)
    parser.add_argument('--candidate', required=True)
    args=parser.parse_args()
    campaign=args.campaign.resolve()
    plan=json.loads((campaign/'plan.json').read_text())
    if plan.get('controller_backward_mode'):
        from aldp_scaled_release import require_scaled_release
        require_scaled_release(plan, args.stage, args.candidate)
    source_sha=verify_source()
    if plan.get('controller_backward_mode') and source_sha != plan['source_manifest_sha256']:
        raise RuntimeError('Registered LV numerical source changed')
    for name, expected in plan['frozen_inputs'].items():
        if sha256(name) != expected: raise RuntimeError('Frozen input changed: '+name)
    if not torch.cuda.is_available(): raise RuntimeError('Remote GPU required')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    cfg, output=configuration(plan,campaign,args.stage,args.candidate)
    if output.exists() and any(output.iterdir()): raise RuntimeError('Preserve old output; new attempt required')
    observer = None
    try:
        from tr_asbs.process.chart_source import load_calibration_bundle
        observer=GridObserver(load_calibration_bundle(cfg.potential.calibration_bundle_path),output,cfg,plan,source_sha,args.stage,args.candidate)
        cost=expected_cost(cfg,observer.phase_count)
        train_case(dict(plan,max_openmm_per_geometry_run=cost['total_openmm_configurations']),
            campaign,args.stage,'low_density',0,source_sha,config_and_output=(cfg,output),observer=observer,
            provenance=dict(candidate=args.candidate,plan_sha256=sha256(campaign/'plan.json'),backend='torch'),
            module_setup=lambda module: setattr(observer, 'training_module', module))
        verify(output,plan,source_sha,args.stage,args.candidate)
        print('postflight_verified=true',flush=True)
    except Exception as error:
        module = getattr(observer, 'training_module', None)
        measured = None if module is None else dict(
            OpenMM_configurations=int(module.potential.aldp_potential.openmm_configurations_evaluated),
            optimizer_steps=int(module.global_step))
        write_json(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),source_sha256=source_sha,
            measured_failed_attempt_cost=measured))
        raise


if __name__=='__main__': main()
