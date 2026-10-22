import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import mdtraj
from pathlib import Path
from .bgflow_energy import bgflowEnergy
from .base_energy import BaseEnergy
from tr_asbs.fab.target_distributions.aldp import AldpBoltzmann
from tr_asbs.fab.trajectory import load_trajectory_frames, trajectory_metadata
from .internal_coordinate_chart import InternalCoordinateChart
from tr_asbs.utils.md_utils import compute_all_dihedrals, compute_torsion_klds, compute_joint_W2, plot_ramachandran_AD, plot_marginal_group_AD, plot_energy_hist, to_md_traj
from tr_asbs.utils.distribution_distance import compute_distribution_distances, w2_1d_samples

# Helper functions for the ALDP target model configuration
def make_aldp_model(config, device, energy_mode='regularized'):
    transform_mode = config['system'].get('transform', 'mixed')
    shift_dih = config['system'].get('shift_dih', False)
    env = config['system'].get('env', 'vacuum')
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    
    target = AldpBoltzmann(
        data_path=config['data']['transform'],
        temperature=config['system']['temperature'],
        energy_cut=config['system']['energy_cut'],
        energy_max=config['system']['energy_max'],
        n_threads=config['system']['n_threads'],
        transform=transform_mode,
        ind_circ_dih=ind_circ_dih,
        shift_dih=shift_dih,
        env=env,
        device=device,
        transform_max_frames=config['data'].get('transform_max_frames'),
        transform_stride=config['data'].get('transform_stride', 1),
        transform_offset=config['data'].get('transform_offset', 0),
        energy_mode=energy_mode,
    )
    return target.to(device)

def load_config(path):
    import yaml
    with open(path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)


def validate_calibration_energy_definition(bundle, energy_definition):
    definition = bundle.get('energy_definition')
    if definition is None and energy_definition['mode'] != 'regularized':
        raise ValueError('Physical energy requires a versioned calibration energy bank')
    if definition is not None and definition != energy_definition:
        raise ValueError('Frozen calibration energy definition mismatch')
    
def filter_chirality(x, ind=[17, 26], mean_diff=-0.043, threshold=0.8):
    """
    Filters batch for the L-form
    :param x: Input batch
    :param ind: Indices to be used for determining the chirality
    :param mean_diff: Mean of the difference of the coordinates
    :param threshold: Threshold to be used for splitting
    :return: Returns indices of batch, where L-form is present
    """
    diff_ = torch.column_stack((x[:, ind[0]] - x[:, ind[1]],
                                x[:, ind[0]] - x[:, ind[1]] + 2 * np.pi,
                                x[:, ind[0]] - x[:, ind[1]] - 2 * np.pi))
    min_diff_ind = torch.min(torch.abs(diff_), 1).indices
    diff = diff_[torch.arange(x.shape[0]), min_diff_ind]
    ind = torch.abs(diff - mean_diff) < threshold
    return ind

class AlanineDipeptidePotential(bgflowEnergy):
    """
    Internal Potential class for ALDP to match the bgflowEnergy interface, 
    similar to LennardJonesPotential.
    """
    def __init__(self, device="cpu", transform_data_path=None,
                 transform_max_frames=4096, transform_stride=1, transform_offset=0,
                 energy_mode='regularized'):
        # ALDP usually has 22 atoms in 3D (dim=66)
        super().__init__(dim=66)
        
        repo_dir = Path(__file__).resolve().parent.parent.parent
        config = load_config(repo_dir / 'tr_asbs/fab/config.yaml')
        # position_min_energy.pt (a single reference minimum-energy structure)
        # was never present in this repo -- val_split_AD.h5 (a full REMD
        # trajectory, see aldp.py's data_path handling for the "h5" branch) is
        # supported directly for initializing the internal-coordinate (Z-matrix)
        # transform and is what we actually have on disk.
        config['data']['transform'] = str(transform_data_path or repo_dir.parent / "data/alanine_dipeptide/val.h5")
        config['data'].update(transform_max_frames=transform_max_frames,
                              transform_stride=transform_stride, transform_offset=transform_offset)
        self.transform_metadata = trajectory_metadata(config['data']['transform'],
            transform_max_frames, transform_stride, transform_offset)
        self.system_metadata = config['system'].copy()
        
        self.p = make_aldp_model(config, device, energy_mode=energy_mode)
        self.energy_definition = dict(schema='aldp_energy_definition_v1',
            mode=self.p.energy_mode, units='kT', gas_constant_kj_mol_kelvin=8.314e-3,
            temperature=config['system']['temperature'])
        if energy_mode == 'regularized':
            self.energy_definition.update(energy_cut=config['system']['energy_cut'],
                                          energy_max=config['system']['energy_max'])
        self.p.float() # Ensure float32
        self.openmm_configurations_evaluated = 0
        self.openmm_batch_calls = 0

    def _energy(self, z):
        # The FAB AldpBoltzmann.log_prob usually returns log probability
        return -self._log_prob(z)[:, None]

    def _log_prob(self, z):
        self.openmm_configurations_evaluated += z.shape[0]
        self.openmm_batch_calls += 1
        return self.p.log_prob(z)
    
    def x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.p.coordinate_transform.inverse(x)
        return z

    def z_to_x(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.p.coordinate_transform.forward(z)
        return x
    
    def energy_x(self, x: torch.Tensor) -> torch.Tensor:
        self.openmm_configurations_evaluated += x.shape[0]
        self.openmm_batch_calls += 1
        return self.p.p.norm_energy(x)

class AlanineDipeptideEnergyIC(BaseEnergy):
    def __init__(
        self,
        dim,
        ref_samples_path,
        n_particles=22,
        spatial_dim=3,
        device="cpu",
        transform_data_path=None,
        transform_max_frames=4096,
        transform_stride=1,
        transform_offset=0,
        reference_max_frames=4096,
        reference_stride=1,
        reference_offset=0,
        coordinate_chart="legacy",
        calibration_bundle_path=None,
        chart_key="current",
        energy_mode="regularized",
    ):
        # Matches LennardJonesEnergy signature
        if dim not in (60, 66):
            raise ValueError("ALDP has 60 internal or 66 Cartesian coordinates")
        super().__init__(60)
        self.cartesian_dim = n_particles*spatial_dim
        self.device = device
        self.aldp_potential = AlanineDipeptidePotential(device=device,
            transform_data_path=transform_data_path or ref_samples_path,
            transform_max_frames=transform_max_frames, transform_stride=transform_stride,
            transform_offset=transform_offset, energy_mode=energy_mode)
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        if coordinate_chart not in ("legacy", "euclidean"):
            raise ValueError("coordinate_chart must be legacy or euclidean")
        self.coordinate_chart = coordinate_chart
        bundle = None
        if calibration_bundle_path is not None:
            from tr_asbs.process.chart_source import load_calibration_bundle
            bundle = load_calibration_bundle(calibration_bundle_path)
            if coordinate_chart != 'euclidean':
                raise ValueError('Frozen chart calibration requires euclidean coordinates')
            if bundle['system'] != self.aldp_potential.system_metadata:
                raise ValueError('Frozen calibration molecular system mismatch')
            validate_calibration_energy_definition(bundle, self.aldp_potential.energy_definition)
            self.aldp_potential.p.coordinate_transform.load_state_dict(
                bundle['coordinate_transform'], strict=True)
        self.chart = (InternalCoordinateChart(self.aldp_potential.p.coordinate_transform.transform)
                      if coordinate_chart == "euclidean" else None)
        if bundle is not None:
            self.chart.set_dihedral_offsets(bundle['chart_offsets'][chart_key])
        assert self.cartesian_dim == 66, "Dimension mismatch"
        self.ref_samples = load_trajectory_frames(ref_samples_path, reference_max_frames,
                                                 reference_stride, reference_offset)
        self.data_metadata = dict(transform=self.aldp_potential.transform_metadata,
            reference=trajectory_metadata(ref_samples_path, reference_max_frames,
                                          reference_stride, reference_offset))
        self.data_metadata['system'] = self.aldp_potential.system_metadata
        self.data_metadata['energy_definition'] = self.aldp_potential.energy_definition
        if bundle is not None:
            import hashlib
            self.data_metadata['frozen_calibration'] = dict(
                path=str(Path(calibration_bundle_path).resolve()),
                sha256=hashlib.sha256(Path(calibration_bundle_path).read_bytes()).hexdigest(),
                chart_key=chart_key, data=bundle['provenance'])
        self.data_metadata['geometry'] = dict(native_dim=60, cartesian_dim=66,
            coordinate_chart=coordinate_chart, chart_support_enforced=self.chart is not None,
            kernel_state="unconstrained_y" if self.chart is not None else "legacy_unwrapped_z",
            periodic_kernel_implemented=False,
            scientific_status="not_evaluated")
        print("Loaded reference samples from:", ref_samples_path, "size:", len(self.ref_samples))
        # self.ref_samples = torch.from_numpy(data.xyz).reshape(-1, dim).float().to(device)
        
    def filter_samples(self, samples: torch.Tensor) -> torch.Tensor:
        # samples are in internal coordinate 'z' space
        with torch.no_grad():
            x_samples = self.native_to_cartesian(samples)
            ind = filter_chirality(x_samples, ind=[17, 26], mean_diff=-0.043, threshold=0.8)
        return ind

    def eval(self, samples: torch.Tensor) -> torch.Tensor:
        # samples here are expected to be in the 'z' (internal coordinate) space.
        # .to(self.device) is a no-op (and stays autograd-differentiable) when
        # samples is already on the right device -- needed because log_details
        # passes in CPU-detached snapshots while the coordinate transform's own
        # buffers live on self.device.
        samples = samples.view(samples.shape[0], -1).to(self.device)
        if samples.shape[-1] != 60:
            raise ValueError("eval expects 60 internal coordinates")
        if self.chart is not None:
            z, log_jac = self.chart(samples)
            return -self.aldp_potential._log_prob(z)-log_jac
        return -self.aldp_potential._log_prob(samples)

    def native_to_cartesian(self, samples):
        samples = samples.to(self.device)
        z = self.chart(samples)[0] if self.chart is not None else samples
        return self.aldp_potential.z_to_x(z)

    def cartesian_to_native(self, samples):
        z = self.aldp_potential.x_to_z(samples.to(self.device))
        return self.chart.inverse(z, canonical=True) if self.chart is not None else z
    
    def eval_x(self, x: torch.tensor):
        return self.aldp_potential.energy_x(x)

    @torch.no_grad()
    def sample_loss(self, samples: torch.Tensor) -> torch.Tensor:
        """Override BaseEnergy.sample_loss: `samples` (data_1, the SDE's own
        output) is in z (internal-coordinate) space, but self.sample() returns
        ground truth in x (Cartesian) space -- BaseEnergy's generic version
        assumes both are in the same space and compares 60-dim vs 66-dim
        tensors directly, which crashes. Convert samples to x-space first."""
        gt_samples = self.sample(samples.shape[0], coordinate="x")
        x_samples = self.native_to_cartesian(samples.view(samples.shape[0], -1))
        x_samples = x_samples.view(x_samples.shape[0], -1).detach().cpu()
        gt_samples = gt_samples.view(gt_samples.shape[0], -1).detach().cpu()
        return compute_distribution_distances(x_samples, gt_samples.to(x_samples.device))

    @torch.no_grad()
    def energy_dist_error(self, energies: torch.Tensor) -> torch.Tensor:
        """Override BaseEnergy.energy_dist_error: ground-truth samples from
        self.sample() are in x-space, but self.eval() expects z-space input --
        compare both energies in z space, including the coordinate Jacobian."""
        gt_samples = self.sample(energies.shape[0], coordinate="native")
        gt_energies = -self.eval(gt_samples)
        energy_w2 = w2_1d_samples(energies, gt_energies).detach().cpu()
        return {"energy_W_2": energy_w2}

    def to(self, device):
        self.device = device
        self.aldp_potential.to(device)
        if self.chart is not None:
            self.chart.to(device)
        return self
        
    def sample(self, size: int | tuple, return_tensor=True, coordinate="native") -> torch.Tensor:
        num_samples = size if isinstance(size, int) else int(np.prod(size))
        if coordinate not in ("x", "z", "native"):
            raise ValueError("coordinate must be x, z, or native")
        idxs = torch.randint(len(self.ref_samples), (num_samples,))
        gt_samples = self.ref_samples[idxs]
        
        if return_tensor:
            x = torch.from_numpy(gt_samples.xyz.reshape(-1, self.cartesian_dim)).float().to(self.device)
            if coordinate == "x":
                return x
            return self.aldp_potential.x_to_z(x) if coordinate == "z" else self.cartesian_to_native(x)
        
        return gt_samples

    def __repr__(self):
        return f"ALDP-{self.n_particles}-D{self.spatial_dim}"

    def visualize(self, samples, title=None):
        x = self.native_to_cartesian(samples.reshape(-1, 60))
        return plot_ramachandran_AD(x, n_particles=self.n_particles, spatial_dim=self.spatial_dim)

    def visualize_raw(self, samples, title=None):
        return self.visualize(samples, title=title)
    
    
    def log_details(self, results: dict, prefix="", visualize=True) -> dict:
        # results["data_1"] is deliberately CPU-resident (log_details builds
        # `results` with .detach().cpu() throughout, to avoid pinning GPU
        # memory for logging), but the coordinate transform's own buffers
        # (mean_b1 etc.) live on self.device -- z_to_x needs both on the same
        # device.
        gen_samples_z = results["data_1"].to(self.device)  # coordinate z
        gen_samples = self.native_to_cartesian(gen_samples_z)
        ref_samples = self.sample(results["data_1"].shape[0], coordinate="x")
        
        logs = super().log_details(results, prefix=prefix, visualize=False)
        
        gen_dihedrals = compute_all_dihedrals(gen_samples, n_particles=self.n_particles, spatial_dim=self.spatial_dim)
        ref_dihedrals = compute_all_dihedrals(ref_samples, n_particles=self.n_particles, spatial_dim=self.spatial_dim)

        metrics, gen_hists, ref_hists = compute_torsion_klds(gen_dihedrals, ref_dihedrals)
        w2_phi_psi = compute_joint_W2(gen_samples, ref_samples, n_particles=self.n_particles,
                                    spatial_dim=self.spatial_dim, periodic=True)
        
        logs.update({f"{prefix}loss/{k}": v for k, v in metrics.items()})
        logs.update({f"{prefix}loss/w2_phi_psi_torus": w2_phi_psi})
        logs[f"{prefix}loss/native_energy_W_2"] = logs[f"{prefix}loss/energy_W_2"]
        gen_physical_energy = self.eval_x(gen_samples).detach().cpu()
        ref_physical_energy = self.eval_x(ref_samples).detach().cpu()
        logs[f"{prefix}loss/physical_energy_W_2"] = float(w2_1d_samples(gen_physical_energy, ref_physical_energy))
        logs.update({f"{prefix}cost/openmm_configurations": self.aldp_potential.openmm_configurations_evaluated,
                     f"{prefix}cost/openmm_batch_calls": self.aldp_potential.openmm_batch_calls})
        
        # 2. プロット作成
        if visualize:
            img_ramachandran = plot_ramachandran_AD(gen_samples, n_particles=self.n_particles, spatial_dim=self.spatial_dim)
            img_marginals = plot_marginal_group_AD(gen_hists, ref_hists)
            energy_figs = plot_energy_hist(gen_physical_energy, ref_physical_energy)
            logs.update({**img_ramachandran, **img_marginals, **energy_figs})

        return logs
