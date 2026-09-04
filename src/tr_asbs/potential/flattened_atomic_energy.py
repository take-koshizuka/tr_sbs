"""Adapts a graph-batch (ThermoAtomicData) BasePotential -- e.g. Ala2's
SumPotential(OpenMMPotential + metadynamics + chirality restraints) -- to
the flat-tensor BaseEnergy interface used by the flat-tensor gamma/TR-SBS
pipeline (train_gamma_asbs_iw.GammaASBSModule), so it can be driven by a
plain FourierMLP controller/corrector instead of PaiNN. Precedent: same
"chemically distinct atoms, no permutation-equivariance needed" rationale
as mace_butane.yaml (see tr_asbs.potential.mace_off_energy), just applied
to a physics-based (OpenMM) rather than learned (MACE) potential.

OpenMM's own energy computation is an external, non-autograd C++ call, so
grad_E uses its analytic forces directly -- exactly like
AtomicPotentialGradientMixin does for the graph-batch pipeline -- rather
than torch.autograd.grad on eval().
"""

import ase.units as units
import mdtraj
import torch

from tr_asbs.data.atomic_data import ThermoAtomicData
from tr_asbs.potential.base_energy import BaseEnergy
from tr_asbs.potential.base_potential import BasePotential
from tr_asbs.utils.composition import composition_to_atomic_numbers
from tr_asbs.utils.geometry import subtract_com_vector
from tr_asbs.utils.md_utils import (
    compute_all_dihedrals,
    compute_torsion_klds,
    plot_marginal_group_AD,
    plot_ramachandran_AD,
)


class FlattenedAtomicPotential(BaseEnergy):
    def __init__(
        self,
        atomic_potential: BasePotential,
        n_particles: int,
        composition: str,
        temperature: float,
        ref_samples_path: str,
        spatial_dim: int = 3,
        pbc: list = [False, False, False],
        cell=None,
        device="cpu",
    ):
        super().__init__(n_particles * spatial_dim)
        self.atomic_potential = atomic_potential
        self.atomic_potential.eval()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.device = torch.device(device)
        self.temperature = float(temperature)

        atomic_numbers = composition_to_atomic_numbers(composition)
        assert len(atomic_numbers) == n_particles, (
            f"composition has {len(atomic_numbers)} atoms, expected n_particles={n_particles}"
        )
        self.atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long)
        self.pbc_t = torch.as_tensor(pbc, dtype=torch.bool)
        self.cell_t = (
            torch.zeros(3, 3, dtype=torch.float) if cell is None else torch.as_tensor(cell, dtype=torch.float)
        )

        # Reference MD trajectory: same file used as the graph-batch
        # pipeline's eval_traj (mdtraj xyz is in nanometers; OpenMMPotential
        # -- and therefore this whole flat wrapper -- operates in Angstrom,
        # matching the graph-batch pipeline's own data.pos convention).
        ref_traj = mdtraj.load(ref_samples_path)
        ref_xyz_angstrom = ref_traj.xyz.reshape(-1, n_particles, spatial_dim) * 10.0
        ref_xyz_angstrom = subtract_com_vector(ref_xyz_angstrom, n_particles, spatial_dim)
        self.ref_samples = torch.from_numpy(ref_xyz_angstrom).float()

    @property
    def is_mean_free(self) -> bool:
        return True

    def to(self, device):
        self.device = torch.device(device)
        self.atomic_potential.to(device)
        return self

    def _to_atomic_data(self, x: torch.Tensor) -> ThermoAtomicData:
        pos = x.reshape(x.shape[0], self.n_particles, self.spatial_dim).detach()
        data_list = [
            ThermoAtomicData.from_geometry(
                pos=pos[b],
                atomic_numbers=self.atomic_numbers,
                cell=self.cell_t.unsqueeze(0),
                pbc=self.pbc_t.unsqueeze(0),
                temperature=torch.as_tensor([self.temperature], dtype=torch.float),
            )
            for b in range(pos.shape[0])
        ]
        # atomic_potential's own submodules (metadynamics/chirality-restraint
        # buffers included) live on self.device -- callers may pass x on a
        # different device (e.g. log_details' CPU-resident data_1, kept off
        # GPU to avoid pinning memory for logging), so build the batch on
        # self.device explicitly rather than inheriting x's device.
        return ThermoAtomicData.from_data_list(data_list).to(self.device)

    @torch.no_grad()
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """E(x). x: (B, n_particles*3) or (B, n_particles, 3), in Angstrom."""
        data = self._to_atomic_data(x)
        out = self.atomic_potential(data)
        return out["energy"].to(x)

    @torch.no_grad()
    def grad_E(self, x: torch.Tensor) -> torch.Tensor:
        """∇E(x) via OpenMM's own analytic forces (not autograd -- OpenMM's
        energy computation is an external, non-differentiable C++ call)."""
        orig_shape = x.shape
        data = self._to_atomic_data(x)
        out = self.atomic_potential(data)
        forces = out["forces"].reshape(x.shape[0], self.n_particles, self.spatial_dim)
        grad = -forces / (units.kB * self.temperature)
        return grad.reshape(orig_shape).to(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -self.eval(x)

    def sample(self, size: int | tuple) -> torch.Tensor:
        num_samples = size if isinstance(size, int) else sum(list(size))
        num_ref = len(self.ref_samples)
        if num_samples <= num_ref:
            idxs = torch.randperm(num_ref)[:num_samples]
        else:
            # ala2_0.1ns.pdb has only 1000 frames, fewer than eval_samples
            # (2000 by default) -- sample with replacement rather than
            # crashing (randperm(num_ref)[:num_samples] would silently
            # return only num_ref indices, breaking the caller's expected
            # reshape).
            idxs = torch.randint(0, num_ref, (num_samples,))
        gt_samples = self.ref_samples[idxs]
        sample_shape = size if isinstance(size, tuple) else (size,)
        return gt_samples.reshape(sample_shape + (self.n_particles, self.spatial_dim)).to(self.device)

    @torch.no_grad()
    def visualize(self, samples, show=True, title=None):
        import matplotlib.pyplot as plt
        import wandb

        fig, ax = plt.subplots(figsize=(6, 4))
        energy_samples = self.eval(samples).detach().cpu()
        ax.hist(energy_samples, bins=100, density=True, alpha=0.6, histtype="step", linewidth=4, label="generated data")
        ax.set_xlabel("Energy (eV)")
        ax.legend()
        wb_key = f"figures/vis_{title}" if title is not None else "figures/vis"
        wb = {wb_key: wandb.Image(fig)}
        plt.close(fig)
        return wb

    def log_details(self, results: dict, prefix: str = "", visualize: bool = True) -> dict:
        data_1 = results["data_1"]
        logs = super().log_details(results, prefix=prefix, visualize=False)

        gen_samples = data_1.reshape(data_1.shape[0], -1).detach().cpu()
        # Use the FULL reference pool (not a size-matched self.sample() draw)
        # for the marginal/KLD histograms -- ala2_0.1ns.pdb has only 1000
        # frames, fewer than eval_samples (2000 by default), so a
        # size-matched draw would resample WITH replacement and be a fresh,
        # noisy random subsample every single call (unlike the atomic/PaiNN
        # pipeline's eval_traj-based marginal plot, which always uses the
        # full, fixed reference trajectory deterministically). Histogram
        # comparison doesn't require equal N between gen and ref, so there's
        # no downside to using all available reference frames here.
        ref_samples = self.ref_samples.reshape(self.ref_samples.shape[0], -1).detach().cpu()

        gen_dihedrals = compute_all_dihedrals(gen_samples, n_particles=self.n_particles, spatial_dim=self.spatial_dim)
        ref_dihedrals = compute_all_dihedrals(ref_samples, n_particles=self.n_particles, spatial_dim=self.spatial_dim)
        metrics, gen_hists, ref_hists = compute_torsion_klds(gen_dihedrals, ref_dihedrals)
        logs.update({f"{prefix}loss/{k}": v for k, v in metrics.items()})

        if visualize:
            logs.update(plot_ramachandran_AD(gen_samples, n_particles=self.n_particles, spatial_dim=self.spatial_dim, ref_samples=ref_samples))
            logs.update(plot_marginal_group_AD(gen_hists, ref_hists))

        return logs

    def __repr__(self):
        return f"FlattenedAtomic-{self.n_particles}atoms"
