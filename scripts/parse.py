import argparse
import collections
import logging
from typing import List, Tuple, Union

import sklearn.model_selection
from rdkit.Chem import Mol

from graphdg import tools
from graphdg.parse import iso17, qm9
from graphdg.parse.tools import select_parsable_atoms
from graphdg.tools import mol_to_smiles, save_pickle


def filter_warnings() -> None:
    # Hide RDKit warnings when generating molecular graphs
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Command line tool for dataset preparation')

    # Dataset
    parser.add_argument('--path', dest='data_path', help='path to dataset', type=str, required=True)
    parser.add_argument('--name', help='dataset name', required=False, default=None)
    parser.add_argument('--seed', help='seed for train test split', type=int, default=0)
    parser.add_argument('--data_type',
                        help='kind of dataset',
                        type=str,
                        required=False,
                        choices=['iso17', 'qm9'],
                        default='iso17')
    parser.add_argument('--test_size',
                        help='If between 0.0 and 1.0, represents the proportion of the dataset '
                        'to include in the test split. If > 1, represents the absolute number of test samples.',
                        type=float,
                        required=False,
                        default=0.15)
    parser.add_argument('--min_num_samples',
                        help='minimum number of samples per molecule',
                        type=int,
                        default=3,
                        required=False)

    # Logging
    parser.add_argument('--log_level', help='log level', type=str, default='INFO')

    return parser.parse_args()


def get_config() -> dict:
    config = vars(parse_args())
    if not config['name']:
        config['name'] = config['data_type']
    return config


def train_test_split(molecules: List[Mol], test_size: Union[float, int], seed: int) -> Tuple[List[Mol], List[Mol]]:
    smiles_list = [mol_to_smiles(mol) for mol in molecules]
    smiles_set = list(set(smiles_list))

    train_smiles, test_smiles = sklearn.model_selection.train_test_split(
        smiles_set,
        test_size=test_size,
        random_state=seed,
    )

    logging.info(f'Number of distinct molecules: {len(train_smiles)} train, {len(test_smiles)} test')

    train = []
    test = []
    for smiles, molecule in zip(smiles_list, molecules):
        if smiles in train_smiles:
            train.append(molecule)
        else:
            test.append(molecule)

    logging.info(f'Dataset sizes: {len(train)} train, {len(test)} test')

    return train, test


def filter_molecules(molecules: List[Mol], min_num_samples: int) -> List[Mol]:
    smiles_mols_dict = collections.defaultdict(list)
    for mol in molecules:
        smiles_mols_dict[mol_to_smiles(mol)].append(mol)

    selection = []
    for key, mols in smiles_mols_dict.items():
        if len(mols) >= min_num_samples:
            selection += mols

    return selection


def main() -> None:
    config = get_config()

    tag = f"{config['name']}_split-{config['seed']}"
    suffix = '.pkl'
    tools.setup_logger(config['log_level'])

    # Load structures
    if config['data_type'] == 'iso17':
        atoms_list = iso17.parse_dataset(path=config['data_path'])
    else:
        atoms_list = qm9.parse_dataset(path=config['data_path'])

    # Convert to RDKit molecules
    molecules = select_parsable_atoms(atoms_list)

    # Minimum number of samples
    if config['min_num_samples']:
        molecules = filter_molecules(molecules, min_num_samples=config['min_num_samples'])

    # Train-Test split
    test_size = config['test_size'] if config['test_size'] < 1 else int(config['test_size'])
    train_mols, test_mols = train_test_split(molecules, test_size=test_size, seed=config['seed'])

    # Save datasets
    save_pickle(train_mols, path=tag + '_train' + suffix)
    save_pickle(test_mols, path=tag + '_test' + suffix)


if __name__ == '__main__':
    filter_warnings()
    main()
