# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import wandb


def _plot_ramachandran_row(axes, traj, row_label, n_residues, nbins):
    phi_indices, phi_rad = md.compute_phi(traj)
    psi_indices, psi_rad = md.compute_psi(traj)
    phi = np.rad2deg(phi_rad)
    psi = np.rad2deg(psi_rad)
    for i in range(n_residues):
        ax = axes[i]
        h = ax.hist2d(
            phi[:, i],
            psi[:, i],
            bins=nbins,
            range=[[-180, 180], [-180, 180]],
            cmap="viridis",
            density=True,
            cmin=1e-6,
        )
        ax.set_title(f"Residue {i + 1} Density", fontsize=12, fontweight="bold")
        if row_label is not None:
            ax.set_ylabel(f"{row_label}\n" + r"$\psi$ (degrees)") if i == 0 else None
        elif i == 0:
            ax.set_ylabel(r"$\psi$ (degrees)")
        if i != 0:
            ax.set_yticklabels([])
        ax.set_xlabel(r"$\phi$ (degrees)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.axhline(0, color="white", linestyle="--", alpha=0.5)
        ax.axvline(0, color="white", linestyle="--", alpha=0.5)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        plt.colorbar(h[3], ax=ax, shrink=0.8)


def make_ramachandran_figure(traj: md.Trajectory, ref_traj: md.Trajectory = None, nbins: int = 100, gen_run_id: str = None):
    """2D phi/psi Ramachandran density plot, one panel per residue.

    If `ref_traj` is given, a second row is added below the generated
    trajectory's row showing the same plot for the reference (MD) data, so
    the two can be visually compared side by side. `gen_run_id` (e.g. the
    source W&B run id) is appended to the "generated" row's label.
    """
    phi_indices, phi_rad = md.compute_phi(traj)
    psi_indices, psi_rad = md.compute_psi(traj)
    if phi_rad.size == 0 or psi_rad.size == 0:
        return None

    n_residues = phi_rad.shape[1]
    n_rows = 2 if ref_traj is not None else 1
    fig, axes = plt.subplots(
        n_rows, n_residues,
        figsize=(5.0 * n_residues, 4.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    gen_label = "generated" if ref_traj is not None else None
    if gen_label is not None and gen_run_id is not None:
        gen_label = f"generated ({gen_run_id})"
    _plot_ramachandran_row(axes[0], traj, gen_label, n_residues, nbins)
    if ref_traj is not None:
        _plot_ramachandran_row(axes[1], ref_traj, "reference", n_residues, nbins)

    return fig


def periodic_convolution(x, kernel):
    x_padded = np.concatenate([x, x, x])
    y_padded = np.convolve(x_padded, kernel, mode='same')
    return y_padded[x.size:-x.size]


def compute_torsion_histogram(
    traj: md.Trajectory,
    Z_indices: np.ndarray,
    smooth: bool = False,
    nbins: int = 200,
    hist_range: list = [-np.pi, np.pi]
):
    # compute the dihedrals
    torsions = md.compute_dihedrals(traj, Z_indices)

    torsion_histogram = []
    for i in range(torsions.shape[1]):
        htrain, e = np.histogram(torsions[:, i], nbins, range=hist_range, density=True)
        if smooth:
            # smooth the histogram with a Gaussian kernel
            smooth_kernel = [0.25, 0.5, 1.0, 0.5, 0.25]  # optional
            htrain = periodic_convolution(htrain, smooth_kernel)
        torsion_histogram.append(htrain)

    return torsion_histogram


def decompose_Z_indices(cart_indices, Z_indices):
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
    cart_atom_indices = np.array([6, 8, 9, 10, 14])
    Z_indices_no_order = np.array([[0, 1, 4, 6],
                [1, 4, 6, 8],
                [2, 1, 4, 0],
                [3, 1, 4, 0],
                [4, 6, 8, 14],
                [5, 4, 6, 8],
                [7, 6, 8, 4],
                [11, 10, 8, 6],
                [12, 10, 8, 11],
                [13, 10, 8, 11],
                [15, 14, 8, 16],
                [16, 14, 8, 6],
                [17, 16, 14, 15],
                [18, 16, 14, 8],
                [19, 18, 16, 14],
                [20, 18, 16, 19],
                [21, 18, 16, 19]])

    cart_atom_indices = np.array(cart_atom_indices)
    batchwise_Z_indices, _ = decompose_Z_indices(cart_atom_indices, Z_indices_no_order)
    Z_indices = np.vstack(batchwise_Z_indices)
    return Z_indices


# adapted from SNF
def plot_marginal_group_AD(
    gen_traj: md.Trajectory,
    ref_traj: md.Trajectory,
    smooth: bool = False,
    ref_torsion_hist=None,
):
    plt.close()
    # plot config
    nbins = 200
    hist_range = [-np.pi, np.pi]
    torsions_complex = [0, 1, 2, 10, 12]

    # compute the histogram
    Z_indices = get_Z_indices()
    if ref_torsion_hist is None:
        ref_torsion_hist = compute_torsion_histogram(ref_traj, Z_indices, smooth, nbins, hist_range)
    gen_torsion_hist = compute_torsion_histogram(gen_traj, Z_indices, smooth, nbins, hist_range)

    # plot the histogram
    fig, axes = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(15, 3))
    fig.subplots_adjust(hspace=0.05, wspace=0.15)
    titles = ['$\\phi$', '$\\gamma_1$', '$\\psi$', '$\\gamma_2$', '$\\gamma_3$']
    for ax, torsion_idx, title in zip(axes, torsions_complex, titles):
        xticks = np.linspace(*hist_range, nbins)
        ax.plot(xticks, ref_torsion_hist[torsion_idx], color='grey', linewidth=3, label='ref')
        ax.plot(xticks, gen_torsion_hist[torsion_idx], color='blue', linewidth=3, label='gen')

        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi))
        ax.set_xticklabels(('$-\\pi$', '$-\\pi/2$', 0, '$\\pi/2$', '$\\pi$'))

        ax.set_ylim(0, 1)
        ax.set_yticks((0, .25, .5, .75, 1))

        ax.set_title(title, fontsize=15)
        ax.grid(True)

    axes[0].set_ylabel('density', fontsize=15)
    axes[-1].legend(loc='upper right', fontsize=10)

    # compute KL divergence
    EPS = 1e-5
    torsion_klds = []
    for torsion_idx in torsions_complex:
        ref_hist = ref_torsion_hist[torsion_idx]
        gen_hist = gen_torsion_hist[torsion_idx]

        kld_unscaled = np.sum(ref_hist * np.log((ref_hist + EPS) / (gen_hist + EPS)))
        kld = kld_unscaled * (hist_range[1] - hist_range[0]) / nbins
        torsion_klds.append(kld)

    result = {
        "Vis/marginal": wandb.Image(fig),
        "eval/kld_phi": torsion_klds[0],
        "eval/kld_gamma1": torsion_klds[1],
        "eval/kld_psi": torsion_klds[2],
        "eval/kld_gamma2": torsion_klds[3],
        "eval/kld_gamma3": torsion_klds[4],
    }
    plt.close(fig)
    return result
