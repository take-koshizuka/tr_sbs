import torch
import numpy as np
import mdtraj
from mdtraj.core.trajectory import Trajectory
from openmmtools.testsystems import AlanineDipeptideVacuum
import matplotlib as mpl
from matplotlib import pyplot as plt
from PIL import Image
import wandb
import ot  # POT: Python Optimal Transport

# ==========================================
# 1. ユーティリティ・定数
# ==========================================

# 二面角計算用のインデックス定義 (Alanine Dipeptide用)
# indices_in_z_matrix: [phi, gamma1, psi, gamma2, gamma3] に対応するZ-matrix内行番号
TORSION_INDICES_MAP = {
    "indices": [0, 1, 2, 10, 12],
    "labels": [r'$\phi$', r'$\gamma_1$', r'$\psi$', r'$\gamma_2$', r'$\gamma_3$'],
    "keys": ["phi", "gamma1", "psi", "gamma2", "gamma3"]
}

def fig2img(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buf, mode="RGBA")

def to_md_traj(samples: torch.Tensor | Trajectory, n_particles: int, spatial_dim: int) -> Trajectory:
    if isinstance(samples, Trajectory):
        return samples
    
    # torch.Tensor (N, 66) or (N, 22, 3) -> Trajectory
    aldp = AlanineDipeptideVacuum()
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    coords = samples.detach().cpu().numpy().reshape(-1, n_particles, spatial_dim)
    return mdtraj.Trajectory(coords, topology)

# ==========================================
# 2. 角度計算ロジック
# ==========================================
def decompose_Z_indices(cart_indices, Z_indices):
    import numpy as np
    known_indices = cart_indices
    Z_placed = np.zeros(Z_indices.shape[0])
    Z_indices_decomposed = []
    while np.count_nonzero(Z_placed) < Z_indices.shape[0]:
        Z_indices_cur = []
        for i in range(Z_indices.shape[0]):
            if not Z_placed[i] and np.all([Z_indices[i, j] in known_indices for j in range(1, 4)]):
                Z_indices_cur.append(Z_indices[i])
                Z_placed[i] = 1
        Z_indices_cur = np.array(Z_indices_cur)
        known_indices = np.concatenate([known_indices, Z_indices_cur[:, 0]])
        Z_indices_decomposed.append(Z_indices_cur)

    index2order = np.concatenate([cart_indices] + [Z[:, 0] for Z in Z_indices_decomposed])

    return Z_indices_decomposed, index2order

def get_Z_indices():
    """Z-matrixの原子インデックス定義を返す"""
    cart_atom_indices = np.array([6, 8, 9, 10, 14])
    z_indices_raw = np.array([
        [0, 1, 4, 6], [1, 4, 6, 8], [2, 1, 4, 0], [3, 1, 4, 0],
        [4, 6, 8, 14], [5, 4, 6, 8], [7, 6, 8, 4], [11, 10, 8, 6],
        [12, 10, 8, 11], [13, 10, 8, 11], [15, 14, 8, 16], [16, 14, 8, 6],
        [17, 16, 14, 15], [18, 16, 14, 8], [19, 18, 16, 14], [20, 18, 16, 19], [21, 18, 16, 19]
    ])
    # cart_indices = np.concatenate([[i*3, i*3+1, i*3+2] for i in cart_atom_indices])
    batchwise_Z_indices, _ = decompose_Z_indices(cart_atom_indices, z_indices_raw)
    Z_indices = np.vstack(batchwise_Z_indices)
    return Z_indices

def compute_all_dihedrals(samples: torch.Tensor | Trajectory, n_particles: int, spatial_dim: int) -> torch.Tensor:
    """すべての二面角を一括計算する"""
    traj = to_md_traj(samples, n_particles=n_particles, spatial_dim=spatial_dim)
    z_indices = get_Z_indices()
    # mdtraj.compute_dihedrals は (n_frames, n_dihedrals) を返す
    dihedrals = mdtraj.compute_dihedrals(traj, z_indices)
    return torch.from_numpy(dihedrals)

def extract_phi_psi(all_dihedrals: torch.Tensor) -> torch.Tensor:
    """一括計算された二面角から phi(0) と psi(2) を抽出する"""
    # TORSION_INDICES_MAP["indices"] の 0番目と2番目
    phi = all_dihedrals[:, 0]
    psi = all_dihedrals[:, 2]
    return torch.stack([phi, psi], dim=1)

# ==========================================
# 3. 統計・評価指標
# ==========================================

def compute_torsion_histograms(dihedrals: torch.Tensor, nbins: int = 200, hist_range=[-np.pi, np.pi]):
    """各二面角のヒストグラムを計算する"""
    hists = []
    for i in range(dihedrals.shape[1]):
        h, _ = np.histogram(dihedrals[:, i].numpy(), bins=nbins, range=hist_range, density=True)
        hists.append(h)
    return np.array(hists)

def compute_torsion_klds(gen_dihedrals: torch.Tensor, ref_dihedrals: torch.Tensor, nbins: int = 200):
    """KLダイバージェンスのみを計算する"""
    hist_range = [-np.pi, np.pi]
    gen_hists = compute_torsion_histograms(gen_dihedrals, nbins, hist_range)
    ref_hists = compute_torsion_histograms(ref_dihedrals, nbins, hist_range)
    
    EPS = 1e-5
    results = {}
    bin_width = (hist_range[1] - hist_range[0]) / nbins
    
    for i, key in enumerate(TORSION_INDICES_MAP["keys"]):
        idx = TORSION_INDICES_MAP["indices"][i]
        ref_h, gen_h = ref_hists[idx], gen_hists[idx]
        kld = np.sum(ref_h * np.log((ref_h + EPS) / (gen_h + EPS))) * bin_width
        results[f"kld_{key}"] = kld
    return results, gen_hists, ref_hists

def compute_joint_W2(gen_samples, ref_samples, n_particles: int, spatial_dim: int) -> float:
    """Phi-Psi 空間での Wasserstein-2 距離"""
    # angles計算を統合
    gen_angles = extract_phi_psi(compute_all_dihedrals(gen_samples, n_particles=n_particles, spatial_dim=spatial_dim))
    ref_angles = extract_phi_psi(compute_all_dihedrals(ref_samples, n_particles=n_particles, spatial_dim=spatial_dim))
    
    M = torch.cdist(gen_angles, ref_angles)
    n, m = M.shape
    # ot.emd2 は平方コストを受け取るため M**2
    w2_sq = ot.emd2(torch.ones(n)/n, torch.ones(m)/m, M**2)
    return float(torch.sqrt(w2_sq))

# ==========================================
# 4. 可視化関数
# ==========================================
def plot_ramachandran_AD(samples: torch.Tensor, n_particles: int, spatial_dim: int, ref_samples: torch.Tensor = None, weights=None, title: str = None, nbins: int = 64, gen_run_id: str = None) -> dict:
    """2D phi/psi Ramachandran histogram for `samples` (generated). If
    `ref_samples` is given, a second panel with the reference (MD) data is
    plotted alongside it for a direct visual comparison. `gen_run_id` (e.g.
    the source W&B run id) is appended to the "generated" panel's title."""
    angles = extract_phi_psi(compute_all_dihedrals(samples, n_particles=n_particles, spatial_dim=spatial_dim))
    phi, psi = angles[:, 0].numpy(), angles[:, 1].numpy()
    bins = [np.linspace(-np.pi, np.pi, nbins), np.linspace(-np.pi, np.pi, nbins)]

    ncols = 2 if ref_samples is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
    axes = axes[0]

    gen_title = "generated" if ref_samples is not None else None
    if gen_title is not None and gen_run_id is not None:
        gen_title = f"generated ({gen_run_id})"

    axes[0].hist2d(phi, psi, bins=bins, norm=mpl.colors.LogNorm(), weights=weights)
    axes[0].set_title(gen_title)
    axes[0].set_xlabel(r'$\phi$')
    axes[0].set_ylabel(r'$\psi$')
    axes[0].set_xlim(-np.pi, np.pi)
    axes[0].set_ylim(-np.pi, np.pi)

    if ref_samples is not None:
        ref_angles = extract_phi_psi(compute_all_dihedrals(ref_samples, n_particles=n_particles, spatial_dim=spatial_dim))
        ref_phi, ref_psi = ref_angles[:, 0].numpy(), ref_angles[:, 1].numpy()
        axes[1].hist2d(ref_phi, ref_psi, bins=bins, norm=mpl.colors.LogNorm())
        axes[1].set_title("reference")
        axes[1].set_xlabel(r'$\phi$')
        axes[1].set_xlim(-np.pi, np.pi)
        axes[1].set_ylim(-np.pi, np.pi)

    PIL_im = fig2img(fig)
    plt.close(fig)

    key = f"figures/ramachandran_{title}" if title else "figures/ramachandran"
    return {key: wandb.Image(PIL_im)}

def plot_marginal_group_AD(gen_hists, ref_hists, title: str = None) -> dict:
    """計算済みのヒストグラムからプロットのみ生成する"""
    hist_range = [-np.pi, np.pi]
    nbins = gen_hists.shape[1]
    xticks = np.linspace(*hist_range, nbins)
    
    indices = TORSION_INDICES_MAP["indices"]
    labels = TORSION_INDICES_MAP["labels"]

    fig, axes = plt.subplots(1, 5, sharey=True, sharex=True, figsize=(15, 3))
    for ax, idx, label in zip(axes, indices, labels):
        ax.plot(xticks, ref_hists[idx], color='grey', linewidth=2, label='ref')
        ax.plot(xticks, gen_hists[idx], color='blue', linewidth=2, label='gen')
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    PIL_im = fig2img(fig)
    plt.close(fig)

    key = f"figures/marginal_{title}" if title else "figures/marginal"
    return {key: wandb.Image(PIL_im)}

def plot_energy_hist(gen_energies, ref_energies, title: str = None) -> dict:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist(ref_energies, bins=30, alpha=0.5, label='ref', color='blue', density=True)
    ax.hist(gen_energies, bins=30, alpha=0.5, label='gen', color='orange', density=True)
    ax.legend()
    
    PIL_im = fig2img(fig)
    plt.close(fig)
    key = f"figures/energy_hist_{title}" if title else "figures/energy_hist"
    return {key: wandb.Image(PIL_im)}


# ====================================================
# ====== copy from boltzgen/openmm_interface.py ======
# ====================================================


# Gas constant in kJ / mol / K
R = 8.314e-3


class OpenMMEnergyInterface(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, openmm_context, temperature):
        device = input.device
        n_batch = input.shape[0]
        input = input.view(n_batch, -1, 3)
        n_dim = input.shape[1]
        energies = torch.zeros((n_batch, 1), dtype=input.dtype)
        forces = torch.zeros_like(input)

        kBT = R * temperature
        input = input.cpu().detach().numpy()
        for i in range(n_batch):
            # reshape the coordinates and send to OpenMM
            x = input[i, :].reshape(-1, 3)
            # Handle nans and infinities
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                energies[i, 0] = np.nan
            else:
                openmm_context.setPositions(x)
                state = openmm_context.getState(getForces=True, getEnergy=True)

                # get energy
                energies[i, 0] = (
                    state.getPotentialEnergy().value_in_unit(
                        unit.kilojoule / unit.mole) / kBT
                )

                # get forces
                f = (
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule / unit.mole / unit.nanometer
                    )
                    / kBT
                )
                forces[i, :] = torch.from_numpy(-f)
        forces = forces.view(n_batch, n_dim * 3)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output, None, None


class OpenMMEnergyInterfaceParallel(torch.autograd.Function):
    """
    Uses parallel processing to get the energies of the batch of states
    """
    @staticmethod
    def var_init(sys, temp):
        """
        Method to initialize temperature and openmm context for workers
        of multiprocessing pool
        """
        global temperature, openmm_context
        temperature = temp
        sim = app.Simulation(sys.topology, sys.system,
                             mm.LangevinIntegrator(temp * unit.kelvin,
                                                   1.0 / unit.picosecond,
                                                   1.0 * unit.femtosecond),
                             platform=mm.Platform.getPlatformByName('Reference'))
        openmm_context = sim.context

    @staticmethod
    def batch_proc(input):
        # Process state
        # openmm context and temperature are passed a global variables
        input = input.reshape(-1, 3)
        n_dim = input.shape[0]

        kBT = R * temperature
        # Handle nans and infinities
        if np.any(np.isnan(input)) or np.any(np.isinf(input)):
            energy = np.nan
            force = np.zeros_like(input)
        else:
            openmm_context.setPositions(input)
            state = openmm_context.getState(getForces=True, getEnergy=True)

            # get energy
            energy = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule / unit.mole) / kBT

            # get forces
            force = -state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule / unit.mole / unit.nanometer) / kBT
        force = force.reshape(n_dim * 3)
        return energy, force

    @staticmethod
    def forward(ctx, input, pool):
        device = input.device
        input_np = input.cpu().detach().numpy()
        energies_out, forces_out = zip(*pool.map(
            OpenMMEnergyInterfaceParallel.batch_proc, input_np))
        energies_np = np.array(energies_out)[:, None]
        forces_np = np.array(forces_out)
        energies = torch.from_numpy(energies_np)
        forces = torch.from_numpy(forces_np)
        energies = energies.type(input.dtype)
        forces = forces.type(input.dtype)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output, None, None


def regularize_energy(energy, energy_cut, energy_max):
    # Cast inputs to same type
    energy_cut = energy_cut.type(energy.type())
    energy_max = energy_max.type(energy.type())
    # Check whether energy finite
    energy_finite = torch.isfinite(energy)
    # Cap the energy at energy_max
    energy = torch.where(energy < energy_max, energy, energy_max)
    # Make it logarithmic above energy cut and linear below
    energy = torch.where(
        energy < energy_cut, energy, torch.log(energy - energy_cut + 1) + energy_cut
    )
    energy = torch.where(energy_finite, energy,
                         torch.tensor(np.nan, dtype=energy.dtype, device=energy.device))
    return energy


# ====================================================
# ======== copy from boltzgen/distribution.py ========
# ====================================================

class Boltzmann:
    """
    Boltzmann distribution using OpenMM to get energy and forces
    """
    def __init__(self, sim_context, temperature, energy_cut, energy_max):
        """
        Constructor
        :param sim_context: Context of the simulation object used for energy
        and force calculation
        :param temperature: Temperature of System
        """
        # Save input parameters
        self.sim_context = sim_context
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)

        # Set up functions
        self.openmm_energy = OpenMMEnergyInterface.apply
        self.regularize_energy = regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.sim_context, temperature)[:, 0],
            self.energy_cut, self.energy_max)

    def log_prob(self, z):
        return -self.norm_energy(z)


class BoltzmannParallel:
    """
    Boltzmann distribution using OpenMM to get energy and forces and processes the
    batch of states in parallel
    """
    def __init__(self, system, temperature, energy_cut, energy_max, n_threads=None):
        """
        Constructor
        :param system: Molecular system
        :param temperature: Temperature of System
        :param energy_cut: Energy at which logarithm is applied
        :param energy_max: Maximum energy
        :param n_threads: Number of threads to use to process batches, set
        to the number of cpus if None
        """
        # Save input parameters
        self.system = system
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)
        self.n_threads = mp.cpu_count() if n_threads is None else n_threads

        # Create pool for parallel processing
        self.pool = mp.Pool(self.n_threads, OpenMMEnergyInterfaceParallel.var_init,
                            (system, temperature))

        # Set up functions
        self.openmm_energy = OpenMMEnergyInterfaceParallel.apply
        self.regularize_energy = regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.pool)[:, 0],
            self.energy_cut, self.energy_max)

    def log_prob(self, z):
        return -self.norm_energy(z)
