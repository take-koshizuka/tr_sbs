import math
import random
from typing import List
import numpy as np

import sys
import traceback
import numpy as np

import torch

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from ase import Atoms
from ase.io import read

from mace.tools import torch_geometric
from mace.tools.utils import AtomicNumberTable
from mace.data.utils import config_from_atoms
from mace.data import AtomicData

from src.utils.graph_utils import subtract_com_batch
from hydra.utils import to_absolute_path


# Define covalent radii (in Å) for relevant elements
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "O": 0.66,
    "N": 0.71,
    "F": 0.57,
    "Cl": 0.99,
    "Br": 1.14,
    "S": 1.05,
    "I": 1.33,
    "P": 1.07,
}

VAN_DER_WAALS_RADII = {
    "H": 1.10,
    "C": 1.70,
    "O": 1.52,
    "N": 1.55,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.83,
    "S": 1.80,
    "I": 1.98,
    "P": 1.80,
}

# Constants
ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
SYM_LIST = {v: k for k, v in ATOMIC_NUMBERS.items()}


def get_smiles_list(filepath):

    filepath = to_absolute_path(filepath)

    with open(filepath, "r") as f:
        next(f)
        mols = f.readlines()

    num_rot_bonds_list, smiles_list_in, xyz_dir_list_in = zip(
        *[line.strip().split() for line in mols]
    )
    return smiles_list_in


class CenteredRDKit:
    """
    """
    def __init__(
        self,
        smiles_list: List[str] | None = None,
        n_systems_per_batch: int = 1,
        energy = None,
        device = "cpu",
    ) -> None:
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list,]

        assert len(smiles_list) > 0
        assert 0 < n_systems_per_batch <= len(smiles_list)

        self.smiles_list = smiles_list
        self.n_systems_per_batch = n_systems_per_batch
        self.atomic_numbers = energy.atomic_numbers
        self.r_max = energy.r_max
        self.device = device

        if len(smiles_list) == 1:
            self.name = f"rdkit_{smiles_list[0]}"
        else:
            self.name = "rdkit"


    def sample_smiles(self, n_systems: int) -> List[str]:
        """ sample w/o replacement from smiles_list
        """
        assert 0 < n_systems <= len(self.smiles_list)

        # quick handle for single system
        if len(self.smiles_list) == 1:
            return self.smiles_list

        indices = random.sample(range(len(self.smiles_list)), k=n_systems)
        return [self.smiles_list[i] for i in indices]


    def sample(self, shape: tuple) -> torch_geometric.Batch:
        assert len(shape) == 1

        # batch size (B) = n_systems_per_batch (N) * n_conformers_per_sys (M)
        B = shape[0]
        N, M = self.n_systems_per_batch, B // self.n_systems_per_batch
        assert B == N * M

        smiles_list = self.sample_smiles(N)
        assert len(smiles_list) == N

        graph_state_list = []
        for smiles in smiles_list:
            graph_state_list.extend(
                sample_rdkit_graph(
                    smiles,
                    M, #batch_size,
                    self.atomic_numbers,
                    10. * self.r_max, # TODO(ghliu) ensure fully-connected graph
                )
            )
        # assert len(graph_state_list) == B

        # NOTE(ghliu) weird handle to get Batch datatype, should this be ...
        # torch_geometric.Batch.from_data_list(graph_state_list).to(self.device)
        graph_batch_loader = torch_geometric.dataloader.DataLoader(
            dataset=graph_state_list,
            batch_size=len(graph_state_list),
            shuffle=False,
        )
        graph_state = next(iter(graph_batch_loader)).to(self.device)

        # zero center
        graph_state["positions"] = subtract_com_batch(
            graph_state["positions"], graph_state["batch"]
        )
        return graph_state


# NOTE(ghliu): modify from `read_rdkit_mols` without relax
def sample_rdkit_graph(
    smiles,
    batch_size,
    atomic_numbers,
    r_max,
    optimize=False, # NOTE(ghliu)
    # relax=False,
    # calc=None,
    # fmax=0.05,
):
    z_table = AtomicNumberTable([int(z) for z in atomic_numbers])

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to convert SMILES to RDKit molecule: {smiles}", force=True)
        return None
    mol = Chem.AddHs(mol)

    # NOTE(ghliu) for computing `edge_attrs`
    length_matrix, type_matrix, _ = bond_length_matrix(mol)

    # Generate conformers
    seed = np.random.randint(0, np.iinfo(np.int32).max - 1)
    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=batch_size,
        randomSeed=seed,  # for reproducibility NOTE(ghliu)
        pruneRmsThresh=-1,  # Remove similar conformers
        enforceChirality=True,
    )

    if optimize:
        try:
            for conf_id in range(mol.GetNumConformers()):
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        except Exception as e:
            print(f"Error in optimizing {smiles=}!!", file=sys.stderr, force=True)
            print(traceback.format_exc(), file=sys.stderr, force=True)
            raise e


    graph_state_list = []
    for conf in mol.GetConformers():
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conf)
        atom_list = [atom.GetAtomicNum() for atom in new_mol.GetAtoms()]
        positions = conf.GetPositions()
        atoms = Atoms(numbers=atom_list, positions=np.array(positions))

        # ====== NOTE(ghliu) comment out if we need to relax atoms ======
        # if relax:
        #     atoms.calc = calc
        #     opt = LBFGS(atoms)
        #     opt.run(fmax=fmax, steps=1000)
        # ====== NOTE(ghliu) comment out if we need to relax atoms ======

        config = config_from_atoms(atoms=atoms)
        data_i = AtomicData.from_config(
            config, z_table=z_table, cutoff=float(r_max)
        )

        # NOTE(ghliu) add keys `edge_attrs` (for EGNNs) and `smiles`.
        edge_index = data_i["edge_index"]
        length_attr = length_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
        type_attr = type_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
        edge_attr = torch.cat([length_attr, type_attr], dim=-1).float()
        data_i["edge_attrs"] = edge_attr
        data_i["smiles"] = smiles

        graph_state_list.append(data_i)

    if len(graph_state_list) != batch_size:
        msg = "[{}] Query {} conformers but get {}".format(
            smiles, batch_size, len(graph_state_list)
        )
        print(msg, file=sys.stderr, force=True)

    return graph_state_list


def load_crest_graph_state(filename, energy_model):
    loader = xyz_to_loader(filename, energy_model)
    return next(iter(loader))


def get_charge(smiles):
    mol = Chem.MolFromSmiles(smiles)
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return charge


def load_crest_conformer(filename, energy_model, charge):
    return read_xyz_files(
        filename,
        energy_model.atomic_numbers,
        energy_model.r_max,
        charge
    )


# NOTE(ghliu): modify from `generate_conformers` without relax
def generate_conformer(
    graph_state, atomic_numbers, charge,
    # relax=False, calc=None, fmax=0.05, steps=1_000,
):
    atomic_numbers = atomic_numbers.detach().cpu()
    positions = graph_state["positions"].detach().cpu()
    atom_list = graph_state["node_attrs"].detach().cpu()

    # relaxed_positions = []
    # if relax:
    #     ptr = 0
    #     for i in range(len(graph_state["ptr"]) - 1):
    #         num_atoms = graph_state["ptr"][i + 1] - graph_state["ptr"][i]
    #         atom_list_i = atom_list[ptr : ptr + num_atoms]
    #         atomic_number_i = atomic_numbers[torch.nonzero(atom_list_i)[:, 1]]
    #         positions_i = positions[ptr : ptr + num_atoms]
    #         ptr += num_atoms
    #         atoms = Atoms(
    #             numbers=np.array(atomic_number_i), positions=np.array(positions_i)
    #         )
    #         atoms.calc = calc
    #         opt = LBFGS(atoms)
    #         opt.run(fmax=fmax, steps=steps)
    #         relaxed_positions_i = atoms.positions
    #         relaxed_positions.append(torch.from_numpy(relaxed_positions_i))
    #     relaxed_positions = torch.cat(relaxed_positions, dim=0)
    #     positions = relaxed_positions.float()

    gen_mols = []
    ij = 0
    for i in range(len(graph_state["ptr"]) - 1):
        num_atoms = graph_state["ptr"][i + 1] - graph_state["ptr"][i]
        xyz_block = f"{num_atoms.item()}\nGenerated conformer\n"
        for j in range(num_atoms):
            atomic_number = atomic_numbers[torch.nonzero(atom_list[ij])]
            atom_symbol = SYM_LIST[atomic_number[0, 0].item()]
            xyz_block += f"{atom_symbol} {positions[ij, 0].item():.6f} {positions[ij, 1].item():.6f} {positions[ij, 2].item():.6f}\n"
            ij += 1
        try:
            rdkit_mol = Chem.MolFromXYZBlock(xyz_block)
            rdDetermineBonds.DetermineBonds(rdkit_mol, charge=charge)
            if rdkit_mol is not None:
                gen_mols.append(rdkit_mol)
                del rdkit_mol
            else:
                print(f"Failed to convert generated molecule {i}")
        except Exception as e:
            print(f"Error generating molecule {i}: {e}")
            continue
    print("generated {} mols".format(len(gen_mols)))
    return gen_mols


# NOTE(ghliu) remove dependency on smiles.
def xyz_to_loader(filename, energy_model):
    atomic_number_table = energy_model.atomic_numbers
    z_table = AtomicNumberTable([int(z) for z in atomic_number_table])
    r_max = energy_model.r_max
    molecules = read(filename, index=":")
    N = len(molecules)
    # rdmol = Chem.MolFromSmiles(smiles)
    # rdmol = Chem.AddHs(rdmol)
    # bond_matrix, atom_list = bond_length_matrix(rdmol)
    configs = [config_from_atoms(atoms=mol) for mol in molecules]
    data_set = []

    for i, config in enumerate(configs):
        data_i = AtomicData.from_config(
            config, z_table=z_table, cutoff=float(r_max)
        )
        # edge_index = data_i["edge_index"]
        # edge_attr = bond_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
        # data_i["edge_attrs"] = edge_attr
        data_set.append(data_i)

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=data_set,
        batch_size=N,
        shuffle=False,
        drop_last=False,
    )
    return data_loader


# NOTE(ghliu): modify from `read_rdkit_mols` without relax
def sample_rdkit_conformer(
    smiles,
    batch_size,
    optimize=True, # NOTE(ghliu)
    # relax=False,
    # calc=None,
    # fmax=0.05,
    # charge=0,
):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to convert SMILES to RDKit molecule: {smiles}")
        return None
    mol = Chem.AddHs(mol)

    # Generate conformers
    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=batch_size,
        # randomSeed=42,  # for reproducibility NOTE(ghliu)
        pruneRmsThresh=-1,  # Remove similar conformers
        enforceChirality=True,
    )

    if optimize:
        for conf_id in range(mol.GetNumConformers()):
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)

    mol_list = []
    for conf in mol.GetConformers():
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conf)

        # ====== NOTE(ghliu) comment out if we need to relax mols ======
        # atom_list = [atom.GetAtomicNum() for atom in new_mol.GetAtoms()]
        # positions = conf.GetPositions()
        # atoms = Atoms(numbers=atom_list, positions=np.array(positions))
        # if relax:
        #     atoms.calc = calc
        #     opt = LBFGS(atoms)
        #     opt.run(fmax=fmax, steps=1000)

        # if relax:
        #     num_atoms = len(atoms.positions)
        #     xyz_block = f"{num_atoms}\nGenerated conformer\n"
        #     atomic_number = atoms.numbers
        #     positions = atoms.positions
        #     for j in range(num_atoms):
        #         atom_symbol = SYM_LIST[atomic_number[j]]
        #         xyz_block += f"{atom_symbol} {positions[j, 0]:.6f} {positions[j, 1]:.6f} {positions[j, 2]:.6f}\n"
        #     new_mol = Chem.MolFromXYZBlock(xyz_block)
        #     rdDetermineBonds.DetermineBonds(new_mol, charge=charge)
        # ====== NOTE(ghliu) comment out if we need to relax mols ======

        mol_list.append(new_mol)

    return mol_list


######################################################
###### original functions from Adjoint Sampling ######
######################################################

def read_xyz_files(xyz_path, atomic_numbers, r_max, charge):
    z_table = AtomicNumberTable([int(z) for z in atomic_numbers])
    ref_mols = []

    with open(xyz_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and line[0].isdigit():
            num_atoms = int(line)
            energy = float(lines[i + 1])
            positions = []
            atom_list = []

            xyz_block = f"{num_atoms}\n{energy}\n"
            for line in lines[i + 2 : i + 2 + num_atoms]:
                xyz_block += line
                position = np.array(line.split()[1:4], dtype=float)
                positions.append(position)
                atom_symbol = line.split()[0]
                atom_number = ATOMIC_NUMBERS[atom_symbol]
                atom_list.append(atom_number)
            config = config_from_atoms(
                atoms=Atoms(numbers=atom_list, positions=np.array(positions))
            )
            data_i = AtomicData.from_config(
                config, z_table=z_table, cutoff=float(r_max)
            )
            rdkit_mol = Chem.MolFromXYZBlock(xyz_block)
            rdDetermineBonds.DetermineBonds(rdkit_mol, charge=charge)
            if rdkit_mol is not None:
                ref_mols.append(rdkit_mol)
                del data_i, rdkit_mol
            else:
                print(f"Failed to convert conformer at index {i} for {xyz_path}")
            i += num_atoms + 2
        else:
            i += 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ref_mols


def get_covalent_radius(atom_symbol):
    """Returns the covalent radius for a given atom symbol."""
    return COVALENT_RADII.get(atom_symbol, None)


def get_van_der_waals_radius(atom_symbol):
    """Returns the covalent radius for a given atom symbol."""
    return VAN_DER_WAALS_RADII.get(atom_symbol, None)


def bond_length_matrix(rdmol):
    if rdmol is None:
        raise ValueError("Invalid SMILES string.")

    # Get number of atoms (including hydrogens)
    N = rdmol.GetNumAtoms()

    # Initialize the NxN matrix with 'inf' for non-bonded pairs
    length_matrix = torch.full((N, N), fill_value=np.nan)
    length_matrix.fill_diagonal_(0.0)
    type_matrix = torch.full((N, N), fill_value=0)
    # type_matrix.fill_diagonal_(0.0)
    # Create a list to store atom symbols in order
    atomic_numbers = [rdmol.GetAtomWithIdx(i).GetAtomicNum() for i in range(N)]

    # Iterate over bonds in the molecule

    for bond in rdmol.GetBonds():
        # Get the indices of the bonded atoms
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Get the atom symbols for each atom
        atom1, atom2 = (
            rdmol.GetAtomWithIdx(i).GetSymbol(),
            rdmol.GetAtomWithIdx(j).GetSymbol(),
        )

        # Get bond multiplier
        multiplier = 1.5  # * BOND_MULTIPLIERS.get(bond.GetBondType(), 1.0)

        # Get covalent radii and atom-specific tolerances
        # Get radii and tolerances
        radius1, radius2 = get_covalent_radius(atom1), get_covalent_radius(atom2)
        # Calculate the upper limit of bond length based on bond type
        if radius1 is not None and radius2 is not None:
            # check with the chemistry team
            bond_length = (radius1 + radius2) * multiplier
            length_matrix[i, j] = bond_length
            length_matrix[j, i] = bond_length  # Matrix is symmetric

            type_matrix[i, j] = 1
            type_matrix[j, i] = 1

    # fill out non-bond edges with limits
    for i in range(length_matrix.shape[0]):
        for j in range(i, length_matrix.shape[0]):

            if torch.isnan(length_matrix[i, j]):
                atom1, atom2 = (
                    rdmol.GetAtomWithIdx(i).GetSymbol(),
                    rdmol.GetAtomWithIdx(j).GetSymbol(),
                )
                multiplier = (
                    1.0 / 1.5
                )  # (1.2  * BOND_MULTIPLIERS.get(bond.GetBondType(), 1.0) )

                # Get covalent radii and atom-specific tolerances
                # Get radii and tolerances
                radius1, radius2 = get_van_der_waals_radius(
                    atom1
                ), get_van_der_waals_radius(atom2)

                # Calculate the upper limit of bond length based on bond type
                if radius1 is not None and radius2 is not None:
                    # check with the chemistry team
                    bond_length = (radius1 + radius2) * multiplier
                    length_matrix[i, j] = bond_length
                    length_matrix[j, i] = bond_length  # Matrix is symmetric
    # print(matrix)
    return length_matrix, type_matrix, atomic_numbers


def bond_structure_regularizer(
    positions: torch.Tensor,
    bond_limits: torch.Tensor,
    bond_types: torch.Tensor,
    edge_index: torch.Tensor,
    batch_ptr: torch.Tensor,
    alpha: float = 1.0,
):

    bond_norms = torch.sqrt(
        torch.sum(
            (positions[edge_index[1]] - positions[edge_index[0]]) ** 2,
            dim=1,
            keepdim=True,
        )
    )
    bond_mask = bond_types == 1
    no_bond_mask = bond_types == 0

    bond_constraint = torch.nn.functional.relu(bond_mask * (bond_norms - bond_limits))[
        :, 0
    ]
    no_bond_constraint = torch.nn.functional.relu(
        no_bond_mask * (bond_limits - bond_norms)
    )[:, 0]

    constraints = bond_constraint + no_bond_constraint
    input = torch.zeros(len(batch_ptr) - 1).to(positions.device)

    # assuming fully-connected graph
    sys_sizes = batch_ptr[1:] - batch_ptr[:-1]
    n_edges = sys_sizes * (sys_sizes - 1)
    edge_index = torch.arange(n_edges.shape[0]).to(positions.device)
    edge_index = edge_index.repeat_interleave(n_edges)
    input = input.scatter_reduce(0, edge_index, constraints, reduce="sum")

    return input * alpha
