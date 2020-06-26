import json
import logging
import math
import os
import pickle
import sys
from typing import Any, List, Sequence, Iterator, Sized, Tuple, Optional, Union

import numpy as np
from ase import Atom
from rdkit import Chem
from rdkit.Chem import Mol, rdMolAlign

from graphdg.embed import conformer_to_xyz

NODES = 'nodes'
EDGES = 'edges'
RECEIVERS = 'receivers'
SENDERS = 'senders'
GLOBALS = 'globals'
N_NODE = 'n_node'
N_EDGE = 'n_edge'

COORDS = 'coords'
SMILES = 'smiles'


def load_pickle(path: str) -> Any:
    logging.debug(f'Loading object from path {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str):
    logging.debug(f'Saving object to path {path}')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def split_by_edges(data_dicts: List[dict], array: np.ndarray) -> List[np.ndarray]:
    splits = []
    total = 0

    start = 0
    for d in data_dicts:
        total += len(d[EDGES])
        end = start + len(d[EDGES])
        splits.append(array[start:end])
        start = end

    assert (len(array) == total)
    return splits


def get_items(a: Sequence, indices: Sequence) -> List:
    return [a[i] for i in indices]


def get_batches(a: Sized, batch_size: int, shuffle=True, seed: int = None) -> Iterator:
    if shuffle:
        np.random.seed(seed=seed)
        indices = np.random.permutation(range(len(a)))
    else:
        indices = list(range(len(a)))

    assert (batch_size > 0)
    num_batches = math.ceil(len(a) / batch_size)

    for batch in range(num_batches):
        yield indices[batch * batch_size:(batch + 1) * batch_size]


def compute_stats(a: np.ndarray) -> dict:
    return {
        'max_abs': np.max(np.abs(a)),
        'median': np.median(a),
        'median_abs': np.median(np.abs(a)),
        'mean': np.mean(a),
        'mean_abs': np.mean(np.abs(a)),
        'mean_squared': np.mean(np.square(a)),
        'std': np.std(a),
        'se': np.std(a) / np.sqrt(a.size),
    }


def create_directories(directories: List[str]):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_tag(config: dict) -> str:
    return '{exp}_run-{seed}'.format(exp=config['name'], seed=config['seed'])


def save_config(config: dict, file_path: str, verbose=True):
    formatted = json.dumps(config, indent=4, sort_keys=True)

    if verbose:
        logging.info(formatted)

    with open(file=file_path, mode='w') as f:
        f.write(formatted)


def setup_logger(log_level: Union[str, int], file_path: str = None):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if file_path:
        fh = logging.FileHandler(file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def mol_to_smiles(mol: Mol) -> str:
    return Chem.MolToSmiles(mol, allHsExplicit=True)


def to_one_hot(index: int, num_classes: int) -> List[int]:
    array = np.zeros(num_classes, dtype=np.int)
    np.put(array, index, 1)
    return array.tolist()


def select_unique_mols(molecules: List[Mol]) -> List[Mol]:
    unique_tuples: List[Tuple[str, Mol]] = []

    for molecule in molecules:
        duplicate = False
        smiles = mol_to_smiles(molecule)
        for unique_smiles, _ in unique_tuples:
            if smiles == unique_smiles:
                duplicate = True
                break

        if not duplicate:
            unique_tuples.append((smiles, molecule))

    return [mol for smiles, mol in unique_tuples]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class DictSaver:
    def __init__(self, directory: str, tag: str):
        self.directory = directory
        self.tag = tag
        self._suffix = '.json'

    def save(self, d: dict, name: str):
        path = os.path.join(self.directory, self.tag + '_' + name + self._suffix)
        logging.debug(f'Saving dictionary: {path}')
        with open(path, mode='w') as f:
            f.write(json.dumps(d, cls=NumpyEncoder))


def align_conformers(molecule: Mol, heavy_only=True) -> None:
    atom_ids = []
    if heavy_only:
        atom_ids = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    rdMolAlign.AlignMolConformers(molecule, atomIds=atom_ids)


def write_conformers_to_xyz(molecule: Mol, path: str, max_num: Optional[int] = None) -> None:
    num_samples = molecule.GetNumConformers()
    count = num_samples if max_num is None else min(num_samples, max_num)
    string = ''
    for conf_id in range(count):
        string += conformer_to_xyz(molecule, conf_id=conf_id)
        string += '\n'

    with open(path, mode='w') as f:
        f.write(string)


def get_atomic_distance(a0: Atom, a1: Atom) -> float:
    return np.linalg.norm(a0.position - a1.position)
