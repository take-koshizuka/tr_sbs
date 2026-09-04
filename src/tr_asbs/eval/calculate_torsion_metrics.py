# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import mdtraj as md
from scipy.spatial.distance import jensenshannon


def calculate_torsion_metrics(ref_traj, gen_traj, temp=300):
    """
    Computes 2D JSD and Free Energy RMSE between a reference and
    generated trajectory for backbone torsion angles.

    Includes aggregated means across all residues.

    Returns:
        dict: Flat dictionary with keys:
              - 'residue_X/jsd'
              - 'residue_X/rmse_fe'
              - 'mean/jsd'
              - 'mean/rmse_fe'
    """
    results = {}
    jsd_list = []
    rmse_list = []

    # 1. Compute Angles (returns indices and angles in radians)
    _, ref_phi_rad = md.compute_phi(ref_traj)
    _, ref_psi_rad = md.compute_psi(ref_traj)
    _, gen_phi_rad = md.compute_phi(gen_traj)
    _, gen_psi_rad = md.compute_psi(gen_traj)

    # Convert to degrees
    ref_phi = np.rad2deg(ref_phi_rad)
    ref_psi = np.rad2deg(ref_psi_rad)
    gen_phi = np.rad2deg(gen_phi_rad)
    gen_psi = np.rad2deg(gen_psi_rad)

    n_residues = ref_phi.shape[1]

    # Constants
    kT = 0.001987 * temp  # kcal/mol
    epsilon = 1e-10  # Avoid log(0)

    # Define 2D Bins (50x50 grid)
    bins_2d = [np.linspace(-180, 180, 50), np.linspace(-180, 180, 50)]

    # 2. Iterate through residues
    for i in range(n_residues):
        # Compute 2D histograms (Joint Distribution)
        h_ref, _, _ = np.histogram2d(ref_phi[:, i], ref_psi[:, i], bins=bins_2d, density=True)
        h_gen, _, _ = np.histogram2d(gen_phi[:, i], gen_psi[:, i], bins=bins_2d, density=True)

        # --- Metric 1: 2D Jensen-Shannon Divergence ---
        # Flatten to 1D vectors for scipy calculation
        jsd_val = jensenshannon(h_ref.flatten(), h_gen.flatten()) ** 2

        results[f'eval/residue_{i}_jsd'] = jsd_val
        jsd_list.append(jsd_val)

        # --- Metric 2: Free Energy RMSE ---
        # F = -kT * ln(P)
        f_ref = -kT * np.log(h_ref + epsilon)
        f_gen = -kT * np.log(h_gen + epsilon)

        # Mask: Only compare regions populated in the reference (prob > 0.0001)
        # We ignore unpopulated high-energy regions to prevent noise dominance
        mask = h_ref > 1e-4

        if np.sum(mask) > 0:
            mse = np.mean((f_ref[mask] - f_gen[mask]) ** 2)
            rmse_val = np.sqrt(mse)
            results[f'eval/residue_{i}_rmse_fe'] = rmse_val
            rmse_list.append(rmse_val)
        else:
            results[f'eval/residue_{i}_rmse_fe'] = float('nan')

    # 3. Aggregate Metrics
    results['eval/mean_jsd'] = np.mean(jsd_list)

    # Handle NaN values for RMSE aggregation (if any residue failed the mask)
    if rmse_list:
        results['eval/mean_rmse_fe'] = np.nanmean(rmse_list)
    else:
        results['eval/mean_rmse_fe'] = float('nan')

    return results
