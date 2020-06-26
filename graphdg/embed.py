import numpy as np
from rdkit import DistanceGeometry
from rdkit.Chem import Mol
from rdkit.Chem import rdDistGeom

MAX_DISTANCE = 1E3
MIN_DISTANCE = 1E-3


def conformer_to_xyz(molecule: Mol, conf_id=0, comment=None) -> str:
    num_atoms = molecule.GetNumAtoms()
    string = f'{num_atoms}\n'

    if comment:
        string += comment

    conformer = molecule.GetConformer(conf_id)

    for atom_idx in range(molecule.GetNumAtoms()):
        atom = molecule.GetAtomWithIdx(atom_idx)
        position = conformer.GetAtomPosition(atom_idx)
        string += f'\n{atom.GetSymbol()} {position.x} {position.y} {position.z}'

    return string


def get_init_bounds_matrix(mol: Mol) -> np.ndarray:
    num_atoms = mol.GetNumAtoms()
    bounds_matrix = np.zeros(shape=(num_atoms, num_atoms), dtype=np.float)

    # Initial matrix
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            bounds_matrix[i, j] = MAX_DISTANCE
            bounds_matrix[j, i] = MIN_DISTANCE

    return bounds_matrix


def embed_bounds_matrix(mol: Mol, bounds_matrix: np.ndarray, seed: int = 42) -> int:
    DistanceGeometry.DoTriangleSmoothing(bounds_matrix)

    ps = rdDistGeom.EmbedParameters()
    ps.numThreads = 0  # max number of threads supported by the system will be used
    ps.useRandomCoords = True  # recommended for larger molecules
    ps.clearConfs = False
    ps.randomSeed = seed
    ps.SetBoundsMat(bounds_matrix)

    return rdDistGeom.EmbedMolecule(mol, ps)


def embed_conformer(mol: Mol, senders: np.ndarray, receivers: np.ndarray, means: np.ndarray, variances: np.ndarray,
                    seed: int) -> int:
    bounds_matrix = get_init_bounds_matrix(mol)
    means = means.squeeze(-1)

    stds = np.sqrt(variances)
    for sender, receiver, mean, std in zip(senders, receivers, means, stds):
        if sender > receiver:
            continue

        # NB: s < r
        bounds_matrix[sender, receiver] = mean + std
        bounds_matrix[receiver, sender] = np.max([mean - std, MIN_DISTANCE])

    return embed_bounds_matrix(mol, bounds_matrix, seed)
