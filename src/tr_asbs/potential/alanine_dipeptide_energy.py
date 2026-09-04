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
from tr_asbs.utils.md_utils import compute_all_dihedrals, compute_torsion_klds, compute_joint_W2, plot_ramachandran_AD, plot_marginal_group_AD, plot_energy_hist, to_md_traj
from tr_asbs.utils.distribution_distance import compute_distribution_distances, w2_1d_samples

# Helper functions for the ALDP target model configuration
def make_aldp_model(config, device):
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
    )
    return target.to(device)

def load_config(path):
    import yaml
    with open(path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)
    
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
    def __init__(self, device="cpu"):
        # ALDP usually has 22 atoms in 3D (dim=66)
        super().__init__(dim=66)
        
        repo_dir = Path(__file__).resolve().parent.parent.parent
        config = load_config(repo_dir / 'tr_asbs/fab/config.yaml')
        # position_min_energy.pt (a single reference minimum-energy structure)
        # was never present in this repo -- val_split_AD.h5 (a full REMD
        # trajectory, see aldp.py's data_path handling for the "h5" branch) is
        # supported directly for initializing the internal-coordinate (Z-matrix)
        # transform and is what we actually have on disk.
        config['data']['transform'] = str(repo_dir.parent / "data/val_split_AD.h5")
        
        self.p = make_aldp_model(config, device)
        self.p.float() # Ensure float32

    def _energy(self, z):
        # The FAB AldpBoltzmann.log_prob usually returns log probability
        return -self.p.log_prob(z)[:, None]

    def _log_prob(self, z):
        return self.p.log_prob(z)
    
    def x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.p.coordinate_transform.inverse(x)
        return z

    def z_to_x(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.p.coordinate_transform.forward(z)
        return x
    
    def energy_x(self, x: torch.Tensor) -> torch.Tensor:
        return self.p.p.norm_energy(x)

class AlanineDipeptideEnergyIC(BaseEnergy):
    def __init__(
        self,
        dim,
        ref_samples_path,
        n_particles=22,
        spatial_dim=3,
        device="cpu",
    ):
        # Matches LennardJonesEnergy signature
        super().__init__(dim)
        self.device = device
        self.aldp_potential = AlanineDipeptidePotential(device=device)
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        assert dim == n_particles * spatial_dim, "Dimension mismatch"
        self.ref_samples = mdtraj.load(ref_samples_path)
        print("Loaded reference samples from:", ref_samples_path, "size:", len(self.ref_samples))
        # self.ref_samples = torch.from_numpy(data.xyz).reshape(-1, dim).float().to(device)
        
    def filter_samples(self, samples: torch.Tensor) -> torch.Tensor:
        # samples are in internal coordinate 'z' space
        with torch.no_grad():
            x_samples = self.aldp_potential.z_to_x(samples)
            ind = filter_chirality(x_samples, ind=[17, 26], mean_diff=-0.043, threshold=0.8)
        return ind

    def eval(self, samples: torch.Tensor) -> torch.Tensor:
        # samples here are expected to be in the 'z' (internal coordinate) space.
        # .to(self.device) is a no-op (and stays autograd-differentiable) when
        # samples is already on the right device -- needed because log_details
        # passes in CPU-detached snapshots while the coordinate transform's own
        # buffers live on self.device.
        samples = samples.view(samples.shape[0], -1).to(self.device)
        return - self.aldp_potential._log_prob(samples)
    
    def eval_x(self, x: torch.tensor):
        return self.aldp_potential.energy_x(x)

    @torch.no_grad()
    def sample_loss(self, samples: torch.Tensor) -> torch.Tensor:
        """Override BaseEnergy.sample_loss: `samples` (data_1, the SDE's own
        output) is in z (internal-coordinate) space, but self.sample() returns
        ground truth in x (Cartesian) space -- BaseEnergy's generic version
        assumes both are in the same space and compares 60-dim vs 66-dim
        tensors directly, which crashes. Convert samples to x-space first."""
        gt_samples = self.sample(samples.shape[0])
        x_samples = self.aldp_potential.z_to_x(samples.view(samples.shape[0], -1).to(self.device))
        x_samples = x_samples.view(x_samples.shape[0], -1).detach().cpu()
        gt_samples = gt_samples.view(gt_samples.shape[0], -1).detach().cpu()
        return compute_distribution_distances(x_samples, gt_samples.to(x_samples.device))

    @torch.no_grad()
    def energy_dist_error(self, energies: torch.Tensor) -> torch.Tensor:
        """Override BaseEnergy.energy_dist_error: ground-truth samples from
        self.sample() are in x-space, but self.eval() expects z-space input --
        use eval_x (already defined for x-space energies) instead of eval."""
        gt_samples = self.sample(energies.shape[0])
        gt_samples = gt_samples.view(gt_samples.shape[0], -1).detach()
        # -eval_x to match the sign convention `energies` (passed in from
        # log_details as `-self.eval(data_1)`) already uses.
        gt_energies = -self.eval_x(gt_samples)
        energy_w2 = w2_1d_samples(energies, gt_energies).detach().cpu()
        return {"energy_W_2": energy_w2}

    def to(self, device):
        self.device = device
        self.aldp_potential.to(device)
        
    def sample(self, size: int | tuple, return_tensor=True, coordinate="x") -> torch.Tensor:
        num_samples = size if isinstance(size, int) else sum(list(size))
        idxs = torch.randperm(len(self.ref_samples))[:num_samples]
        gt_samples = self.ref_samples[idxs]
        
        if return_tensor:
            return torch.from_numpy(gt_samples.xyz.reshape(-1, self.dim)).float().to(self.device)
        
        return gt_samples

    def __repr__(self):
        return f"ALDP-{self.n_particles}-D{self.spatial_dim}"
    
    
    def log_details(self, results: dict, prefix="", visualize=True) -> dict:
        # results["data_1"] is deliberately CPU-resident (log_details builds
        # `results` with .detach().cpu() throughout, to avoid pinning GPU
        # memory for logging), but the coordinate transform's own buffers
        # (mean_b1 etc.) live on self.device -- z_to_x needs both on the same
        # device.
        gen_samples_z = results["data_1"].to(self.device)  # coordinate z
        gen_samples = self.aldp_potential.z_to_x(gen_samples_z) # coordinate z to x
        ref_samples = self.sample(results["data_1"].shape[0]) # coordinate x
        
        logs = super().log_details(results, prefix=prefix, visualize=False)
        
        gen_dihedrals = compute_all_dihedrals(gen_samples, n_particles=self.n_particles, spatial_dim=self.spatial_dim)
        ref_dihedrals = compute_all_dihedrals(ref_samples, n_particles=self.n_particles, spatial_dim=self.spatial_dim)

        metrics, gen_hists, ref_hists = compute_torsion_klds(gen_dihedrals, ref_dihedrals)
        w2_phi_psi = compute_joint_W2(gen_samples, ref_samples, n_particles=self.n_particles, spatial_dim=self.spatial_dim)
        
        logs.update({f"{prefix}loss/{k}": v for k, v in metrics.items()})
        logs.update({f"{prefix}loss/w2_phi_psi": w2_phi_psi})
        
        # 2. プロット作成
        if visualize:
            img_ramachandran = plot_ramachandran_AD(gen_samples, n_particles=self.n_particles, spatial_dim=self.spatial_dim)
            img_marginals = plot_marginal_group_AD(gen_hists, ref_hists)
            gen_energies = self.eval_x(gen_samples).cpu()
            ref_energies = self.eval_x(ref_samples).cpu()
            energy_figs = plot_energy_hist(gen_energies, ref_energies)
            logs.update({**img_ramachandran, **img_marginals, **energy_figs})

        return logs