
import io
import math
import PIL
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

import networkx as nx

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDetermineBonds


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    # https://stackoverflow.com/a/61756899
    return PIL.Image.frombytes(
        'RGB',
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb()
    )
    # buf = io.BytesIO()
    # fig.savefig(buf)
    # buf.seek(0)
    # img = PIL.Image.open(buf)
    # return img


def interatomic_dist(samples):
    # Compute the pairwise interatomic distances
    # removes duplicates and diagonal
    n_particles = samples.shape[1]
    distances = samples[:, None, :, :] - samples[:, :, None, :]
    distances = distances[
        :,
        torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1,
    ]
    dist = torch.linalg.norm(distances, dim=-1)
    return dist


@torch.no_grad()
def plot_atom_dist_and_energy(
    graph_state,
    E_out,
    E_ref=None,
    min_energy=None,
    max_energy=None,
):
    n_systems = len(graph_state["ptr"]) - 1
    n_particles = int(len(graph_state["batch"]) // n_systems)
    n_spatial_dim = graph_state["positions"].shape[-1]
    x = graph_state["positions"].view(n_systems, n_particles, n_spatial_dim)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    dist_samples = interatomic_dist(x).detach().cpu()

    bins = 200

    axs[0].hist(
        dist_samples.view(-1),
        bins=bins,
        alpha=0.5,
        density=True,
        histtype="step",
        linewidth=4,
    )

    axs[0].set_xlabel("Interatomic distance")
    axs[0].legend(["generated data"])
    energies = E_out["energy"].detach().cpu()
    min_energy = energies.min().item() if min_energy is None else min_energy
    max_energy = energies.max().item() if max_energy is None else max_energy

    axs[1].hist(
        energies,
        bins=100,
        density=True,
        alpha=0.4,
        range=(min_energy, max_energy),
        color="r",
        histtype="step",
        linewidth=4,
        label="generated data",
    )
    if E_ref is not None:
        for x in E_ref["energy"]:
            axs[1].axvline(x=x.cpu(), color="red", linestyle="--")
    axs[1].set_xlabel("Energy")
    axs[1].legend()

    fig.canvas.draw()
    PIL_im = fig2img(fig)
    plt.close()
    return PIL_im


def plot_recall_precision(metrics, smiles, label=None):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    fig.suptitle(f'System: {smiles}', fontsize=13)

    threshold_ranges = metrics["threshold_ranges"]
    axes[0].plot(threshold_ranges, metrics["recall"], linestyle='-', label=label, alpha=0.5, linewidth=2.0)
    axes[1].plot(threshold_ranges, metrics["precision"], linestyle='-', label=label, alpha=0.5, linewidth=2.0)

    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Coverage Recall (%)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Coverage Precision (%)')
    axes[1].legend()
    axes[1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    fig.canvas.draw()
    PIL_im = fig2img(fig)
    plt.close()
    return PIL_im


def to_mol(positions, atom_types):
    """Convert an XYZ file to an RDKit Mol object"""
    #SYM_LIST = {1: "H", 6: "C", 8: "O", 53: "I", 7: "N", 17: "Cl"}
    ATOMIC_NUMBERS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53
    }
    SYM_LIST = {v: k for k, v in ATOMIC_NUMBERS.items()}

    num_atoms = positions.shape[0]
    mol = Chem.Mol()
    edit_mol = Chem.EditableMol(mol)
    conf = Chem.Conformer(num_atoms)
    for i in range(num_atoms):
        atom_symbol = SYM_LIST[atom_types[i]]
        x = positions[i, 0]
        y = positions[i, 1]
        z = positions[i, 2]
        atom = Chem.Atom(atom_symbol)
        atom_idx = edit_mol.AddAtom(atom)
        conf.SetAtomPosition(int(atom_idx), (float(x), float(y), float(z)))

    mol = edit_mol.GetMol()
    mol.AddConformer(conf)
    rdDetermineBonds.DetermineConnectivity(mol)
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=0)
        cm = Chem.RemoveHs(mol)
        smi = Chem.MolToSmiles(cm)
        return mol, smi
    except:
        return mol, "NA"

    # Chem.SanitizeMol(mol)


def plot_conformer(graph_state, outputs, atomic_number_table, n_samples=16):
    n_samples = min(n_samples, len(graph_state["ptr"]) - 1)
    # rows = math.ceil(n_samples / 4)
    # fig, ax = plt.subplots(
    #     rows, 4, figsize=(10 * rows / 4, 10), gridspec_kw={"wspace": 0.5, "hspace": 0.5}
    # )
    # ij = 0
    ptr = graph_state["ptr"]

    mols = []
    smis = []
    for i in range(n_samples):
        indices = torch.nonzero(graph_state["node_attrs"][ptr[i] : ptr[i + 1]])[
            :, 1
        ].int()
        atomic_numbers = (
            atomic_number_table[indices.detach().cpu()].detach().cpu().numpy()
        )
        positions = graph_state["positions"][ptr[i] : ptr[i + 1]].detach().cpu().numpy()
        mol, smi = to_mol(positions, atomic_numbers)
        # mol = AllChem.AddHs(mol)
        # try:
        #     AllChem.EmbedMolecule(mol)
        # except:
        #     print("failed to embed molecule")

        mols.append(mol)
        smis.append(smi)
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(200, 200),
        legends=[smi + '\n\n reg energy:{:.2f}'.format(outputs['reg_energy'][j]) for j, smi  in enumerate(smis)]
    )
    # pimg.save("rdkit_vis.png")
    # img.save("rdkit_vis.png")
    return img


def calc_performance_stats(rmsd_array, threshold, rsmd_array_name: str = ""):
    # Check if array is empty or all NaN
    if rmsd_array.size == 0 or np.all(np.isnan(rmsd_array)):
        msg = f"Warning: Empty or all-NaN RMSD array"
        if rsmd_array_name:
            msg += f", named {rsmd_array_name}"
        print(msg)
        return None

    # Replace inf values with NaN for min operation
    rmsd_array = np.where(np.isinf(rmsd_array), np.nan, rmsd_array)

    coverage_recall = np.nanmean(
        np.nanmin(rmsd_array, axis=1, keepdims=True) < threshold, axis=0
    )
    amr_recall = np.nanmean(np.nanmin(rmsd_array, axis=1))
    coverage_precision = np.nanmean(
        np.nanmin(rmsd_array, axis=0, keepdims=True) < np.expand_dims(threshold, 1),
        axis=1,
    )
    amr_precision = np.nanmean(np.nanmin(rmsd_array, axis=0))

    return coverage_recall, amr_recall, coverage_precision, amr_precision


def get_networkx_graph(mol: Chem.Mol) -> tuple[nx.Graph, dict[int, int]]:
    m = {atom.GetIdx(): atom.GetAtomicNum() for atom in mol.GetAtoms()}
    g = nx.convert.from_edgelist(
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    ).to_undirected()
    g = nx.relabel_nodes(g, m)
    return g, m


def get_map_if_atoms_same_and_undirected_isomorphism(
    mol1: Chem.Mol, mol2: Chem.Mol
) -> list[list[tuple[int, int]]] | None:
    g1, m1 = get_networkx_graph(mol1)
    g2, m2 = get_networkx_graph(mol2)

    if nx.is_isomorphic(g1, g2, node_match=lambda x, y: x == y):
        assert len(m1) == len(m2)
        maps = [[(i, j) for i, j in zip(m1, m2)]]
        return maps  # using rdkit's maps signature. https://github.com/rdkit/rdkit-orig/blob/57058c886a49cc597b0c40641a28697ee3a57aee/rdkit/Chem/AllChem.py#L160
    else:
        return None


def calc_rmsd(gen_mols, ref_mols, only_alignmol=False):
    rmsd_array = np.full((len(ref_mols), len(gen_mols)), np.inf)
    for i, ref_mol in enumerate(tqdm(ref_mols)):
        for j, gen_mol in enumerate(gen_mols):
            ref_mol_noH = Chem.RemoveHs(ref_mol)
            gen_mol_noH = Chem.RemoveHs(gen_mol)

            # only if there are the same number of atoms
            if ref_mol_noH.GetNumAtoms() == gen_mol_noH.GetNumAtoms():
                try:
                    if only_alignmol:
                        rms_dist = AllChem.AlignMol(gen_mol_noH, ref_mol_noH)
                    else:
                        try:
                            # automatically find pairs
                            rms_dist = AllChem.GetBestRMS(gen_mol_noH, ref_mol_noH)
                        except RuntimeError:
                            maps = get_map_if_atoms_same_and_undirected_isomorphism(
                                gen_mol_noH, ref_mol_noH
                            )
                            rms_dist = AllChem.GetBestRMS(
                                gen_mol_noH, ref_mol_noH, map=maps
                            )
                    rmsd_array[i, j] = rms_dist
                except Exception as e:
                    rmsd_array[i, j] = np.nan
            else:
                rmsd_array[i, j] = np.nan
    return rmsd_array


def calculate_rmsd(gen_mols, ref_mols, smiles, threshold_ranges=None):
    if threshold_ranges is None:
        threshold_ranges = np.arange(0, 2.5, 0.125)

    # NOTE(ghliu) copy from eval.py
    only_alignmol = False # TODO(ghliu) check

    # calculate rmsd
    correct_mols = []
    for i, mol in enumerate(gen_mols):
        smi1 = Chem.MolToSmiles(
            Chem.RemoveHs(mol), isomericSmiles=False, canonical=True
        )
        smi2 = Chem.MolToSmiles(
            Chem.RemoveHs(mol), isomericSmiles=True, canonical=True
        )
        if smi1 == smiles or smi2 == smiles:
            correct_mols.append(mol)

    rmsd_array_gen = calc_rmsd(gen_mols, ref_mols, only_alignmol)
    stats_gen_ = calc_performance_stats(
        rmsd_array_gen, threshold_ranges, "gen_mols vs ref_mols"
    )

    rmsd_array_crr = calc_rmsd(correct_mols, ref_mols, only_alignmol)
    stats_crr_ = calc_performance_stats(
        rmsd_array_crr, threshold_ranges, "correct_mols vs ref_mols"
    )

    results = {}

    cr, mr, cp, mp = stats_gen_
    results.update(
        {
            "threshold_ranges": threshold_ranges,
            "recall": cr,
            "precision": cp,
            "mr": mr,
            "mp": mp,
            "samples_correct_smiles": len(correct_mols),
            "samples_valid_rdkit": len(gen_mols),
            # "num_samples": args.num_eval_samples,
        }
    )

    if stats_crr_ is not None:
        cr, mr, cp, mp = stats_crr_
        results.update(
            {"recall_crr": cr, "precision_crr": cp, "mr_crr": mr, "mp_crr": mp}
        )

    return results
